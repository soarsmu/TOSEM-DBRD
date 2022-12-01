"""
We analyze the change json to understand 
what is the percentage of changes happened
"""

import ujson
import argparse

important_fields = {'priority', 'bug_severity', 'component', 'product', 'version', 'short_desc'}
change_num = dict()

def cal_percentage():
    with open(change_json) as f:
        lines = f.readlines()
        
    for line in lines:
        cur_bug = ujson.loads(line)
        for key in cur_bug:
            if key in important_fields:
                if not key in change_num:
                    change_num[key] = 1
                else:
                    change_num[key] += 1
                    
    print(change_num)
    print('======>percentage<======')
    for key in change_num:
        print('--------')
        print(key)
        print(str(100 * float(change_num[key])/float(len(lines))) + '%\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--project', help='project name')

    args = parser.parse_args()
    
    change_json = '../SABD/dataset/{}-initial/{}-initial_changes.json'.format(args.project, args.project)
    
    cal_percentage()