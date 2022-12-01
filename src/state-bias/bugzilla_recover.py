"""
Created on 10 July 2021
recover each bug report's initial info based on history
save it to a seperate file: PROJECT_initial_changes.json
"""

import logging
import copy
from tqdm import tqdm
import codecs
import ujson
import os
from lxml import html
import argparse

from time_map import ActivityItem


def extract_essential_info():
    """
    Extract and save initial_changes.json
    """
    with open(latest_json) as f:
        lines = f.readlines()
    
    bug_ids = set()
    for line in lines:
        cur_br = ujson.loads(line)
        bug_ids.add(cur_br['bug_id'])
        
    changes_list = list()
    
    for file_name in tqdm(os.listdir(data_path)):
        bug_id = file_name.split('.')[0]
        if not bug_id in bug_ids:
            continue
        
        full_file = data_path + file_name
        tree = html.parse(full_file)
        
        changes = dict()
        changes['bug_id'] = str(bug_id)

        try:
            table = tree.findall('//table')[0]
        except IndexError:
            # No change has been made to the bug, e.g., bug_id 555508
            # logging.info('{} no change'.format(bug_id))
            changes_list.append(changes)
            continue

        # sorted by time
        items = []
        for row in table.findall('tr'):
            tds = [td.text_content().strip() for td in row.findall('td')]
            
            if len(tds) < 1:
                continue
            if len(tds) == 5:
                # (who, when) = tds[:2]
                when = tds[1]

            (what, removed, added) = tds[-3:]
            activity_item = ActivityItem(bug_id, when, what.lower(), removed, added)
            items.append(activity_item)

        important_fields = {'priority', 'bug_severity', 'component', 'product', 'version', 'short_desc'}
        
        for item in items:
            if item.what in important_fields:
                changes[item.what] = item.removed
                important_fields.remove(item.what)

        changes_list.append(changes)

    new_list = sorted(changes_list, key=lambda k: int(k['bug_id']))
    
    with codecs.open(initial_changes_json, 'w', encoding = 'utf-8') as jsonFile:
        for item in new_list:
            jsonFile.write(ujson.dumps(item))
            jsonFile.write('\n')


def recover_initial():
    """
    use the latest and initial_changes.json to recover the initial status
    """

    with codecs.open(latest_json, 'r', encoding = 'utf-8') as latest_json_file:
        latest_lines = latest_json_file.readlines()

    with codecs.open(initial_changes_json, 'r', encoding = 'utf-8') as initial_changes_file:
        initial_lines = initial_changes_file.readlines()

    if len(latest_lines) != len(initial_lines):
        logging.info('error, the latest data has different size from changes data')
    else:
        logging.info('same size, starting\n')

    with codecs.open(initial_json, 'w', encoding = 'utf-8') as initial_file:
        initial_list = list()

        for latest_item, change_item in tqdm(zip(latest_lines, initial_lines), total = len(latest_lines)):
            initial_bug = dict()
            initial_bug = copy.deepcopy(ujson.loads(latest_item))

            change_item = ujson.loads(change_item)
            change_item.pop('bug_id')

            for what in change_item:
                initial_bug[what] = copy.deepcopy(change_item[what])

            initial_list.append(initial_bug)

        logging.info('generating completed')
        new_list = sorted(initial_list, key = lambda k: int(k['bug_id']))

        for item in new_list:
            initial_file.write(ujson.dumps(item))
            initial_file.write('\n')

    logging.info('saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process project')
    parser.add_argument('--project', help='project name')

    args = parser.parse_args()

    latest_json = '{}_recent_dec.json'.format(args.project)
    initial_changes_json = '{}_initial_changes.json'.format(args.project)
    
    initial_json = '{}_initial.json'.format(args.project)

    data_path = '/media/data/donggyun/{}_history/'.format(args.project)
    
    extract_essential_info()
    recover_initial()