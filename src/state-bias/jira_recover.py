"""
Recover the Jira state to the initial
Aug 2022
"""

import ujson
from tqdm import tqdm
import random
from datetime import datetime, timezone
import os

state_changes = '../../SABD/dataset/hadoop-initial/spark_hadoop.json'

hadoop = '../../SABD/dataset/hadoop/hadoop.json'
hadoop_changes = 'hadoop_changes_new.json'

spark = '../../SABD/dataset/spark/spark.json'
spark_changes = 'spark_changes_new.json'

hadoop_1day = '../../SABD/dataset/hadoop-1day/hadoop-1day.json'
os.makedirs('../../SABD/dataset/hadoop-1day/', exist_ok=True)
spark_1day = '../../SABD/dataset/spark-1day/spark-1day.json'
os.makedirs('../../SABD/dataset/spark-1day/', exist_ok=True)

def convert_to_utc(created_date):
    dt = datetime.strptime(created_date, '%Y-%m-%dT%H:%M:%S.%f%z').astimezone(tz = timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M:%S %z')

def check_state(project_file):
    hist_ids = set()
    with open(state_changes) as f:
        hist_items = ujson.load(f)
    print('total history brs: {}'.format(len(hist_items)))
    for item in hist_items:
        hist_ids.add(int(item['id']))
    print('total distinguish history brs: {}'.format(len(hist_ids)))
    
    with open(project_file) as f:
        lines_1 = f.readlines()
    print(len(lines_1))
    ids = set()
    no_history = 0
    for line in lines_1:
        cur_br = ujson.loads(line)
        # print(cur_br)
        br_id = int(cur_br['bug_id'])
        ids.add(br_id)
        if not br_id in hist_ids:
            no_history += 1
    print('set: {}'.format(len(ids)))
    print('brs do not have history: {}'.format(no_history))


def count_changes(project_file):
    component_changed = 0
    priority_changed = 0
    version_changed = 0
    short_desc_changed = 0
    long_desc_changed = 0
    with open(project_file) as f:
        lines_1 = f.readlines()
        
    ids = set()
    no_history = 0
    for line in lines_1:
        cur_br = ujson.loads(line)
        # print(cur_br)
        br_id = cur_br['bug_id']
        ids.add(br_id)
    # sample: {'histories': [], 'id': '13348548'}
    with open(state_changes) as f:
        hist_items = ujson.load(f)
    for item in hist_items:
        if item['id'] in ids:
            if len(item['histories']) == 0:
                no_history += 1
                continue
            is_priority_changed, is_component_changed, is_version_changed, \
                is_short_desc_changed, is_long_desc_changed = False, False, \
                    False, False, False
            for activity in item['histories']:
                for change_item in activity['items']:
                    cur_field = change_item['field'].lower()
                    if cur_field == 'priority':
                        is_priority_changed = True
                    if cur_field == 'component':
                        is_component_changed = True
                    if cur_field == 'version':
                        is_version_changed = True
                    if cur_field == 'summary':
                        is_short_desc_changed = True
                    if cur_field == 'description':
                        is_long_desc_changed = True
            if is_priority_changed:
                priority_changed += 1
            if is_component_changed:
                component_changed += 1
            if is_version_changed:
                version_changed += 1
            if is_short_desc_changed:
                short_desc_changed += 1
            if is_long_desc_changed:
                long_desc_changed += 1
    print('Project: {}'.format(project_file.split('/')[-2]))
    print('# of total issues: {}'.format(len(lines_1)))
    print('# of issues without history: {}'.format(no_history))
    print('changed issues percentage %s%%' % (100 * round(((len(lines_1) - no_history) / len(lines_1)), 3)))
    print('priority change: %s%%' % (100 * round((priority_changed / len(lines_1)), 3)))
    print('component change:  %s%%' % (100 * round((component_changed / len(lines_1)), 3)))
    print('version change: %s%%' % (100 * round((version_changed / len(lines_1)), 3)))
    print('short desc change: %s%%' % (100 * round((short_desc_changed / len(lines_1)), 3)))
    print('long desc change: %s%%' % (100 * round((long_desc_changed / len(lines_1)), 3)))
    print('--------------')
    
def generate_initial_json(project_file, target_file):
    with open(project_file) as f:
        lines_1 = f.readlines()
    brs = dict()
    for line in lines_1:
        cur_br = ujson.loads(line)
        # print(cur_br)
        br_id = cur_br['bug_id']
        brs[br_id] = cur_br
    with open(state_changes) as f:
        hist_items = ujson.load(f)

    for item in tqdm(hist_items):
        if item['id'] in brs.keys():
            priority_changed = False
            component_changed = False
            version_changed = False
            short_desc_changed = False
            long_desc_changed = False
            for activity in item['histories']:
                # inside the same activaity, can be changed serveral times
                # e.g., https://issues.apache.org/jira/browse/HADOOP-15693
                # we allow several changes in the same activity
                # the last one would be the initial one
                for change_item in activity['items']:
                    p_changed, c_changed, v_changed, s_changed, l_changed \
                        = False, False, False, False, False
                    cur_field = change_item['field'].lower()
                    if cur_field == 'priority' and priority_changed == False:
                        p_changed = True
                        initial_state = change_item['fromString']
                        if not initial_state:                            
                            initial_state = ''
                        brs[item['id']].update({cur_field: initial_state})
                    if cur_field == 'component' and component_changed == False:
                        c_changed = True
                        initial_state = change_item['fromString']
                        if not initial_state:                            
                            initial_state = ''
                        brs[item['id']].update({cur_field: initial_state})
                    if cur_field == 'version' and version_changed == False:
                        v_changed = True
                        initial_state = change_item['fromString']
                        if not initial_state:
                            initial_state = ''
                        brs[item['id']].update({cur_field: initial_state})
                    if cur_field == 'summary' and short_desc_changed == False:
                        s_changed = True
                        initial_state = change_item['fromString']
                        if not initial_state:
                            initial_state = ''
                        brs[item['id']].update({'short_desc': initial_state})
                    if cur_field == 'description' and long_desc_changed == False:
                        l_changed = True
                        initial_state = change_item['fromString']
                        if not initial_state:
                            initial_state = ''
                        brs[item['id']].update({cur_field: initial_state})
                if p_changed:
                    priority_changed = True
                if c_changed:
                    component_changed = True
                if v_changed:
                    version_changed = True
                if s_changed:
                    short_desc_changed = True
                if l_changed:
                    long_desc_changed = True
    initial_brs = list()
    for k, v in brs.items():
        initial_brs.append(v)
    with open(target_file, 'w') as f:
        for line in initial_brs:
            f.write(ujson.dumps(line) + '\n')

def generate_1day_json(project_file, change_file, target_file):
    with open(project_file) as f:
        lines_1 = f.readlines()
    brs = dict()
    for line in lines_1:
        cur_br = ujson.loads(line)
        # print(cur_br)
        br_id = cur_br['bug_id']
        brs[br_id] = cur_br
    with open(change_file) as f:
        hist_items = ujson.load(f)
    change_count = {'priority': 0, 'component': 0, 'version': 0, 'short_desc': 0, 'long_desc': 0}
    for item in tqdm(hist_items):
        if item['id'] in brs.keys():
            priority_changed = False
            component_changed = False
            version_changed = False
            short_desc_changed = False
            long_desc_changed = False
            cur_creation_time = item['creation_time']
            for activity in item['histories']:
                change_time = convert_to_utc(activity['created'])
                FMT = '%Y-%m-%d %H:%M:%S %z'                
                tdelta = datetime.strptime(change_time, FMT) - datetime.strptime(cur_creation_time, FMT)
                if tdelta.total_seconds() > 86400:
                    break
                # inside the same activaity, can be changed serveral times
                # e.g., https://issues.apache.org/jira/browse/HADOOP-15693
                # we allow several changes in the same activity
                # the last one would be the initial one

                for change_item in activity['items']:
                    cur_field = change_item['field'].lower()
                    if cur_field == 'priority':
                        initial_state = change_item['fromString']
                        if not initial_state:                            
                            initial_state = ''
                        brs[item['id']].update({cur_field: initial_state})
                        priority_changed = True
                    if cur_field == 'component':
                        initial_state = change_item['fromString']
                        if not initial_state:                            
                            initial_state = ''
                        brs[item['id']].update({cur_field: initial_state})
                        component_changed = True
                    if cur_field == 'version':
                        initial_state = change_item['fromString']
                        if not initial_state:
                            initial_state = ''
                        brs[item['id']].update({cur_field: initial_state})
                        version_changed = True
                    if cur_field == 'summary':
                        initial_state = change_item['fromString']
                        if not initial_state:
                            initial_state = ''
                        brs[item['id']].update({'short_desc': initial_state})
                        short_desc_changed = True
                    if cur_field == 'description':
                        initial_state = change_item['fromString']
                        if not initial_state:
                            initial_state = ''
                        brs[item['id']].update({cur_field: initial_state})
                        long_desc_changed = True
            if priority_changed:
                change_count['priority'] += 1
            if component_changed:
                change_count['component'] += 1
            if version_changed:
                change_count['version'] += 1
            if short_desc_changed:
                change_count['short_desc'] += 1
            if long_desc_changed:
                change_count['long_desc'] += 1
                       
    initial_brs = list()
    for k, v in brs.items():
        initial_brs.append(v)
    with open(target_file, 'w') as f:
        for line in initial_brs:
            f.write(ujson.dumps(line) + '\n')
    print(change_count)

def split_changes(project_file, repo_name):
    # randomly check 10 issues
    with open(state_changes) as f:
        hist_items = ujson.load(f)
    with open(project_file) as f:
        lines_1 = f.readlines()
    print(len(lines_1))
    ids = set()
    for line in lines_1:
        cur_br = ujson.loads(line)
        # print(cur_br)
        br_id = cur_br['bug_id']
        ids.add(br_id)
    changes_list = list()
    for item in hist_items:
        if item['id'] in ids:
            changes_list.append(item)
    print(len(changes_list))
    with open('{}_changes.json'.format(repo_name), 'w') as f:
        for line in changes_list:
            f.write(ujson.dumps(line) + '\n')
    
def sanity_check(change_file):
    with open(change_file) as f:
        lines = f.readlines()
    random.shuffle(lines)
    for line in lines[:10]:
        print(line)
        print('--------------')
    
if __name__ == '__main__':
    # check_state(hadoop)
    # check_state(spark)
    # count_changes(hadoop)
    # count_changes(spark)
    # generate_initial_json(hadoop, hadoop_initial)
    # generate_initial_json(spark, spark_initial)
    # split_changes(hadoop, 'hadoop')
    # split_changes(spark, 'spark')
    # sanity_check(hadoop_changes)
    # generate_1day_json('../../SABD/dataset/hadoop/hadoop.json', 'hadoop_changes_new.json', hadoop_1day)
    generate_1day_json('../../SABD/dataset/spark/spark.json', 'spark_changes_new.json', spark_1day)