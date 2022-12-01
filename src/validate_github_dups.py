"""
We sampled 384 issues from all the issues we extracted with
duplicate and validate the successful rate
"""

import ujson
from random import sample
import argparse


def remove_same_ids(project):
    with open('sampled_{}.txt'.format(project)) as f:
        lines = f.readlines()
    for i, line in zip(range(len(lines)), lines):
        bug_id, dup_id = line.strip().split(',')
        if bug_id == dup_id:
            print(i)
            print(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--project', help='project name')

    args = parser.parse_args()
    
    json_file = '../SABD/dataset/{}/{}_latest.json'.format(args.project, args.project)
    
    with open(json_file) as f:
        lines = f.readlines()
    
    dup_bugs = list()
    for line in lines:
        cur_bug = ujson.loads(line)
        if 'dup_id' in cur_bug:
            dup_bugs.append(cur_bug['bug_id'] + ',' + cur_bug['dup_id'])
            
    sampled = sample(dup_bugs, 384)
    
    with open('sampled_{}.txt'.format(args.project), 'w') as f:
        for bug_dup in sampled:
            f.write(bug_dup)
            f.write('\n')
    
    remove_same_ids(args.project)