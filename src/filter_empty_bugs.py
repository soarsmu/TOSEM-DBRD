"""
After getting the soft_cleaned data,
we need to filter out those have both empty
summary and description

"""

import argparse
import ujson

def filter_out():
    final_bug_ids = set()
    with open(soft_cleaned_json) as f:
        lines = f.readlines()
        
    for line in lines:
        cur_bug = ujson.loads(line)
        final_bug_ids.add(cur_bug['bug_id'])
        
    with open(original_json) as f:
        lines = f.readlines()
    
    to_save = list()
    for line in lines:
        cur_bug = ujson.loads(line)
        if cur_bug['bug_id'] in final_bug_ids:
            to_save.append(cur_bug)
            
    with open(original_json, 'w') as f:    
        for line in to_save:
            f.write(ujson.dumps(line))
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--project', help='project name')

    args = parser.parse_args()
    
    soft_cleaned_json = '../SABD/dataset/{}/{}_soft_clean.json'.format(args.project, args.project)
    
    original_json = '../SABD/dataset/{}/{}_original.json'.format(args.project, args.project)
    
    filter_out()