#!/usr/bin/env python3

from pprint import pprint
import ray
import ray._private.test_utils as test_utils
import ray._private.state as state

ray.init()
print('CLUSTER')
pprint(ray.cluster_resources())
print('AVAIL')
pprint(ray.available_resources())
print('TOTAL/NODE')
pprint(state.total_resources_per_node())
print('AVAIL/NODE')
pprint(state.available_resources_per_node())
#print('RESOURCE USAGE')
#pprint(test_utils.get_resource_usage('127.0.0.1:6379'))
