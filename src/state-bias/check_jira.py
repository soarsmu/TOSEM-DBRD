"""
Check the changes in JIRA.
Like mostly happen in 1 hour, 1 day. etc.
"""

import ujson
from datetime import datetime, timezone

def convert_to_utc(created_date):
    dt = datetime.strptime(created_date, '%Y-%m-%dT%H:%M:%S.%f%z').astimezone(tz = timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M:%S %z')

def count_change_period(history_data):
    periods = {'1 hour': 0, '1 day': 0,
        '1 week': 0, 'longer': 0}
    long_desc_changed, short_desc_changed = 0, 0
        
    with open(history_data) as f:
        hist_items = ujson.load(f)
    
    issue_summary_changed, issue_desc_changed = 0, 0
    issue_changed = 0
    total_changes = 0
    for item in hist_items:
        if len(item['histories']) == 0:
            continue
        cur_creation_time = item['creation_time']
        summary_changed = False
        description_changed = False
        for activity in item['histories']:
            total_changes += 1
            change_time = convert_to_utc(activity['created'])
            FMT = '%Y-%m-%d %H:%M:%S %z'
            tdelta = datetime.strptime(change_time, FMT) - datetime.strptime(cur_creation_time, FMT)
            if tdelta.total_seconds() <= 3600:
                periods['1 hour'] += 1
            elif tdelta.total_seconds() <= 86400:
                periods['1 day'] += 1
            elif tdelta.total_seconds() <= 604800:
                periods['1 week'] += 1
            else:
                periods['longer'] += 1
            # else:
            #     issue_changed += 1
            #     periods['longer'] += 1
            #     for change_item in activity['items']:
            #         cur_field = change_item['field'].lower()
            #         if cur_field == 'summary':
            #             short_desc_changed += 1
            #             summary_changed = True
            #         elif cur_field == 'description':
            #             long_desc_changed += 1
            #             description_changed = True
        if summary_changed:
            issue_summary_changed += 1
        if description_changed:
            issue_desc_changed += 1
            
    for key, val in periods.items():
        print('{}: {} ({})'.format(key, val, val / total_changes))
    print(total_changes)
    # print('long desc changes times')
    # print(long_desc_changed)
    # print('short desc changes times')
    # print(short_desc_changed)
    # print('issue summary changed: {}'.format(issue_summary_changed))
    # print('issue description changed: {}'.format(issue_desc_changed))
    # print('total issue changed: {}'.format(issue_changed))
    
def add_creation_time(repo, latest_data, history_data):
    with open (latest_data, 'r') as f:
        latest_data = f.readlines()
    id_creation_date = {}
    for line in latest_data:
        cur_br = ujson.loads(line)
        id_creation_date[cur_br['bug_id']] = cur_br['creation_ts']
        
    with open(history_data) as f:
        hist_items = ujson.load(f)
    
    real_hist_items = []
    for item in hist_items:
        if item['id'] in id_creation_date.keys():
            item['creation_time'] = id_creation_date[item['id']]
            real_hist_items.append(item)
    
    with open('{}_changes_new.json'.format(repo), 'w') as f:
        ujson.dump(real_hist_items, f, indent=4)
    
def count_real_changes(change_file):
    with open(change_file) as f:
        hist_items = ujson.load(f)
    real_changes = 0
    total_changes = 0
    
    for item in hist_items:
        if len(item['histories']) == 0:
            continue
        for activity in item['histories']:
            for change_item in activity['items']:
                total_changes += 1
                before_state = change_item['fromString']
                after_state = change_item['toString']
                if before_state != after_state:
                    real_changes += 1
    print('Total changes: {}'.format(total_changes))
    print('Real changes: {}'.format(real_changes))
    
if __name__ == '__main__':
    # add_creation_time('../../SABD/dataset/hadoop/hadoop.json', 'hadoop_latest_history.json')
    # add_creation_time('spark', '../../SABD/dataset/spark/spark.json', 'spark_hadoop.json')
    # add_creation_time('hadoop', '../../SABD/dataset/hadoop/hadoop.json', 'spark_hadoop.json')
    # print('Spark')
    # count_real_changes('spark_changes_new.json')
    # print('='*10)
    # print('Hadoop')
    count_real_changes('hadoop_changes_new.json')
    count_change_period('hadoop_changes_new.json')
    # count_change_period('hadoop_changes_new.json')