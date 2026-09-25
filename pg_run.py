import argparse
from pprint import pprint
import time

import ray
import ray._private.state as state
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

RUNTIME = 30
NUM_RACKS = 8
NUM_GPU_BUNDLES = 16
ACTORS_PER_BUNDLE = 4
NODES_PER_RACK = (NUM_GPU_BUNDLES * ACTORS_PER_BUNDLE)


@ray.remote(num_cpus=1)
class Creator(object):
    def __init__(self, i, nodes_per_rack):
        self.rack = i
        self.worker_count = nodes_per_rack * ACTORS_PER_BUNDLE
        self.nodes_per_rack = nodes_per_rack
        self.workers = []

    def create(self):
        print("create ", self.rack, ray.get_runtime_context().get_node_id())
        bundles = []
        bundles += [{"CPU": ACTORS_PER_BUNDLE, "GPU": ACTORS_PER_BUNDLE} for _ in range(self.nodes_per_rack)]
        selectors = [{"ray.io/gpu-domain": f"rack-{self.rack}"} for _ in range(self.nodes_per_rack)]
        print(selectors)

        start_ts = time.time()
        pg = placement_group(bundles, bundle_label_selector=selectors, strategy="PACK")
        create_ts = time.time()

        ray.get(pg.ready())
        ready_ts = time.time()

        self.workers = [
            Worker.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)
            ).remote(self.rack, i)
            for i in range(self.worker_count)
        ]
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
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-n', '--nodes-per-rack', type=int, default=NODES_PER_RACK)
    parser.add_argument('-r', '--rack-count', type=int, default=NUM_RACKS)

    args = parser.parse_args()

    ray.init(address="auto")

    creators = []
    '''
    bundles = []
    bundles += [{"CPU": 1} for _ in range(args.rack_count)]
    selectors = [{"ray.io/gpu-domain": f"rack-{i}"} for i in range(args.rack_count)]
    print(selectors)
    '''

    for i in range(args.rack_count):
        bundles = []
        bundles += [{"CPU": 1}]
        selectors = [{"ray.io/gpu-domain": f"rack-{i}"}]
        print(selectors)
        start_ts = time.time()
        pg = placement_group(bundles, bundle_label_selector=selectors, strategy="SPREAD")
        create_ts = time.time()

        ray.get(pg.ready())
        ready_ts = time.time()

        # time.sleep(5)
        print(f"  Creation: {create_ts - start_ts:.2f}s")
        print(f"  Ready:    {ready_ts - create_ts:.2f}s")

        creators.append(
            Creator.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)
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
        if end - start > RUNTIME:
            break


if __name__ == "__main__":
    main()
