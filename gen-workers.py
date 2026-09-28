#!/usr/bin/env python3
import argparse
import os
from jinja2 import Environment, FileSystemLoader


def main():
    ''' main '''
    parser = argparse.ArgumentParser(description='generate rack labels')
    parser.add_argument('-N', '--new-job', action='store_true')
    parser.add_argument('-c', '--copies', type=int, default=1)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-n', '--nodes-per-rack', type=int, default=2)
    parser.add_argument('-p', '--profile', action='store_true')
    parser.add_argument('-r', '--num-racks', type=int, default=8)

    args = parser.parse_args()

    file_loader = FileSystemLoader('.')
    env = Environment(loader=file_loader)

    template = env.get_template('job.yaml.in')

    new = ''
    if args.new_job:
        new = '_new'

    props = {
        'new': new,
        'copies': args.copies,
        'racks': args.num_racks,
        'nodes_per_rack': args.nodes_per_rack,
        'total_nodes': args.num_racks * args.nodes_per_rack,
        'job_racks': args.num_racks * args.copies,
        'job_nodes_per_rack': args.nodes_per_rack * args.copies,
        'job_total_nodes': args.num_racks * args.nodes_per_rack * args.copies,
    }
    if args.profile:
        props['profile'] = '--profile'

    output = template.render(props)

    with open('job.yaml', 'w', encoding='utf-8') as ofile:
        ofile.write(output)
        ofile.write('\n')

    template = env.get_template('RUNIT.in')
    output = template.render(props)

    with open('RUNIT', 'w', encoding='utf-8') as ofile:
        ofile.write(output)
        ofile.write('\n')
    os.chmod('RUNIT', 0o755)

if __name__ == '__main__':
    main()
