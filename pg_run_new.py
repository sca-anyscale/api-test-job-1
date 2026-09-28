import argparse
from contextlib import contextmanager
import copy
from dataclasses import dataclass
from pprint import pprint, pformat
import time
from typing import Dict, List

import ray
import ray._private.state as state
from ray.util.state import list_actors
from ray.util.placement_group import lock_scheduler, unlock_scheduler
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

RUNTIME = 30
NUM_RACKS = 8
NUM_GPU_BUNDLES = 16
ACTORS_PER_BUNDLE = 4
NODES_PER_RACK = (NUM_GPU_BUNDLES * ACTORS_PER_BUNDLE)

@dataclass
class Node:
    node_id: str
    labels: Dict[str, str]
    resources: Dict[str, float]  # look at fixed point here

@dataclass
class Bundle:
    bundle_id: str
    resources: Dict[str, float]  # look at fixed point here
    node_id: str


@contextmanager
def scheduler_lock():
    status = False
    while not status:
        status = lock_scheduler()
    try:
        yield status
    finally:
        unlock_scheduler()


def populate_nodes() -> Dict[str, Node]:
    nodes = ray.nodes()
    resources = state.available_resources_per_node()
    #pprint(resources)
    results: Dict[str, Node] = {}
    for node in nodes:
        results[node['NodeID']] = Node(node_id=node['NodeID'],
                labels=node['Labels'],
                resources=copy.deepcopy(resources[node['NodeID']]),
                )

    #pprint(results)
    return results


def filter_nodes_by_label(nodes: Dict[str, Node], labels: Dict[str, str]) -> List[Node]:
    results: List[Node] = []
    for node in nodes:
        for label in labels:
            if label in nodes[node].labels and nodes[node].labels[label] == labels[label]:
                results.append(nodes[node])

    return results


def allocate_resources_from_node(nodes: List[Node], resources: Dict[str, float]) -> Node:
    for node in nodes:
        if can_allocate(node, resources):
            return node

    return None


def can_allocate(node: Node, resources: Dict[str, float]) -> bool:
    for r in resources:
        if r not in node.resources:
            return False
        if node.resources[r] < resources[r]:
            return False

    for r in resources:
        node.resources[r] -= resources[r]
    return True


@ray.remote(num_cpus=1)
class Creator(object):
    def __init__(self, i, nodes_per_rack):
        self.rack = i
        self.worker_count = nodes_per_rack * ACTORS_PER_BUNDLE
        self.nodes_per_rack = nodes_per_rack
        self.workers = []

    def create(self):
        print("create ", self.rack, ray.get_runtime_context().get_node_id())
        with scheduler_lock() as _:
            base_nodes = populate_nodes()
            nodes = filter_nodes_by_label(base_nodes, {"ray.io/gpu-domain": f"rack-{self.rack}"})
            #pprint(nodes)

            bundles = []
            for i in range(self.worker_count):
                resources = {"CPU": 1.0, "GPU": 1.0}
                node = allocate_resources_from_node(nodes, resources)
                if not node:
                    assert False, "no resources"
                bundles += [Bundle(bundle_id=f"{i}", resources=resources, node_id=node.node_id)]
            #pprint(bundles)

            self.workers = []
            for i in range(self.worker_count):
                #print('SCHED', bundles[i].node_id)
                self.workers.append(
                    Worker.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=bundles[i].node_id, soft=False),
                    ).remote(self.rack, i)
                )
            # self.check()  # XXX

        print("created ", self.rack, ray.get_runtime_context().get_node_id(), len(self.workers))

    def check(self):
        print("check ", self.rack)
        for i in range(self.worker_count):
            ray.get(self.workers[i].work.remote())


@ray.remote(num_gpus=1)
class Worker(object):
    def __init__(self, rack, i):
        self.rack = rack
        self.i = i
        print(f"worker {self.rack}/{self.i} on {ray.get_runtime_context().get_node_id()}")

    def work(self):
        time.sleep(0.2)
        print(f"work {self.rack}/{self.i}")
        # pprint(state.available_resources_per_node())


def main():
    parser = argparse.ArgumentParser(description='new API test')
    parser.add_argument('-R', '--runtime', type=int, default=RUNTIME)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-n', '--nodes-per-rack', type=int, default=NODES_PER_RACK)
    parser.add_argument('-r', '--rack-count', type=int, default=NUM_RACKS)
    parser.add_argument('-s', '--sleep', type=int, default=0)

    args = parser.parse_args()

    ray.init(address="auto")

    creators = []

    job_start = time.time()

    with scheduler_lock() as _:
        base_nodes = populate_nodes()
        for i in range(args.rack_count):
            nodes = filter_nodes_by_label(base_nodes, {"ray.io/gpu-domain": f"rack-{i}"})
            #pprint(nodes)
            bundle = Bundle(bundle_id=f"{i}", resources={"CPU": 1.0}, node_id=nodes[0].node_id)
            #pprint(bundle)
            creators.append(
                Creator.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=bundle.node_id, soft=False),
                ).remote(i, args.nodes_per_rack)
            )

    print(f"Creators: {len(creators)}")
    ray.get([creators[i].create.remote() for i in range(args.rack_count)])
    if args.debug:
        pprint(state.available_resources_per_node())

    start = time.time()
    while True:
        ray.get([creators[i].check.remote() for i in range(args.rack_count)])
        end = time.time()
        if end - start > args.runtime:
            break

    print(f"DONE {time.time() - job_start:.2f}s")
    job_id = ray.get_runtime_context().get_job_id()
    if args.debug:
        pprint(list_actors(filters=[("state", "=", "ALIVE")]))

    filters = [("state", "=", "ALIVE"), ("job_id", "=", f"{job_id}"), ("class_name", "=", "Creator")]
    creators = list_actors(limit=args.rack_count * 2, filters=filters)
    if args.debug:
        print("CREATORS", len(creators))
    assert len(creators) == args.rack_count, "incorrect creator count"
    nodes = set([a.node_id for a in creators])
    assert len(nodes) == args.rack_count, "wrong creator node count"

    filters = [("state", "=", "ALIVE"), ("job_id", "=", f"{job_id}"), ("class_name", "=", "Worker")]
    expected_workers = args.rack_count * args.nodes_per_rack * ACTORS_PER_BUNDLE
    workers = list_actors(limit=expected_workers * 2, filters=filters)
    if args.debug:
        print("WORKERS", len(workers))
    assert len(workers) == expected_workers, "incorrect worker count"
    nodes = set([a.node_id for a in workers])
    assert len(nodes) == args.rack_count * args.nodes_per_rack, "wrong worker node count"

    resources = state.available_resources_per_node()
    for node in nodes:
        if args.debug:
            print(node)
            pprint(resources[node])
        # assert 'GPU' not in resources[node], "unused GPUs"

    if args.sleep:
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
