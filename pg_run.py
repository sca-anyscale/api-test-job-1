import os
import time
import json

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# Tests are supposed to run for 10 minutes.
# RUNTIME = 600
RUNTIME = 60
#NUM_CPU_BUNDLES = 30
#NUM_GPU_BUNDLES = 1


@ray.remote(num_cpus=1)
class Worker(object):
    def __init__(self, i):
        self.i = i

    def work(self):
        time.sleep(0.1)
        print("work ", self.i)


@ray.remote(num_cpus=1, num_gpus=1)
class Trainer(object):
    def __init__(self, i):
        self.i = i

    def train(self):
        time.sleep(0.2)
        print("train ", self.i)


def main():
    ray.init(address="auto")
    res = ray.cluster_resources()
    num_gpu = int(res.get('GPU', 0))
    #num_cpu = int(res['CPU'] - 2 * num_gpu) # assumes 1 GPU per GPU node
    num_cpu = int(res['CPU'] / 2) + 1

    bundles = []
    bundles += [{"CPU": 1, "GPU": 1} for _ in range(num_gpu)]
    bundles += [{"CPU": 1} for _ in range(num_cpu)]

    start_ts = time.time()
    pg = placement_group(bundles, strategy="PACK")
    create_ts = time.time()

    ray.get(pg.ready())
    ready_ts = time.time()

    # time.sleep(5)
    print(f'  Creation: {create_ts - start_ts:.2f}s')
    print(f'  Ready:    {ready_ts - create_ts:.2f}s')

    workers = [
        Worker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)
        ).remote(i)
        for i in range(num_cpu)
    ]

    trainers = [
        Trainer.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)
        ).remote(i)
        for i in range(num_gpu)
    ]

    print(f'Workers: {len(workers)}')
    print(f'Trainers: {len(trainers)}')

    start = time.time()
    while True:
        ray.get([workers[i].work.remote() for i in range(num_cpu)])
        #ray.get(trainer.train.remote())
        ray.get([trainers[i].train.remote() for i in range(num_gpu)])
        end = time.time()
        if end - start > RUNTIME:
            break

    if "TEST_OUTPUT_JSON" in os.environ:
        with open(os.environ["TEST_OUTPUT_JSON"], "w") as out_file:
            results = {}
            json.dump(results, out_file)


if __name__ == "__main__":
    main()
