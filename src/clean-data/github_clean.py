"""
Created on 28 July 2021
modified on 23 Dec 2021

clean the github json files
"""


import logging
import os, sys
import ujson
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='bs4')

import codecs
from tqdm import tqdm
from datetime import datetime, timezone

import re
import argparse


start_date = '2018-01-01 00:00:00 +0000'
end_date = '2020-12-31 23:59:59 +0000'

vscode_address = 'microsoft/vscode'
kibana_address = 'elastic/kibana'

diff_repo = []

def convert_to_utc(created_date):
    # e.g., "2018-07-26T20:32:20Z"
    dt = datetime.strptime(created_date, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M:%S %z')

def extract_from_description(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as f:
        lines = f.readlines()

    jsonFile = codecs.open(latest_json, 'w', encoding = 'utf-8')

    for line in tqdm(lines):
        bug = ujson.loads(line)
        formatted_bug = {}
        formatted_bug['bug_id'] = bug['bug_id']
        # formatted_bug['key'] = bug['key']

        desc_html = bug['description']
        if desc_html == None:
            formatted_bug['description'] = ''
        # remove the html tags

        else:
            try:
                cleantext = BeautifulSoup(desc_html, 'lxml').text
            except TypeError:
                print(bug)
                raise TypeError
            # replace the '\xa0' with ''
            cleantext = cleantext.replace(u'\xa0', u'')

        formatted_bug['creation_ts'] = bug['creation_ts']

        formatted_bug['short_desc'] = bug['short_desc']
        formatted_bug['product'] = ''

        if 'component' in bug:
            formatted_bug['component'] = bug['component']
        else:
            formatted_bug['component'] = ''

        if 'version' in bug:
            formatted_bug['version'] = bug['version']
        else:
            formatted_bug['version'] = ''
        
        formatted_bug['bug_status'] = bug['bug_status']
        # formatted_bug['resolution'] = bug['resolution']

        if 'priority' in bug:
            formatted_bug['priority'] = bug['priority']
        else:
            formatted_bug['priority'] = ''

        if 'bug_severity' in bug:
            formatted_bug['bug_severity'] = bug['bug_severity']
        else:
            formatted_bug['bug_severity'] = ''
        
        formatted_bug['description'] = cleantext
        if 'dup_id' in bug:
            real_dup_id = re.findall(r'\d+', str(bug['dup_id']))
            formatted_bug['dup_id'] = real_dup_id[0]

        jsonFile.write(ujson.dumps(formatted_bug))
        jsonFile.write('\n')


def clean_description(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as f:
        lines = f.readlines()

    jsonFile = codecs.open(latest_json, 'w', encoding = 'utf-8')

    for line in tqdm(lines):
        bug = ujson.loads(line)
        formatted_bug = {}
        formatted_bug['bug_id'] = bug['bug_id']

        desc_html = bug['description']
        if desc_html == None:
            formatted_bug['description'] = ''
        else:
            try:
                cleantext = BeautifulSoup(desc_html, 'lxml').text
            except TypeError:
                print(bug)
                raise TypeError
            # replace the '\xa0' with ''
            cleantext = cleantext.replace(u'\xa0', u'')

        formatted_bug['creation_ts'] = bug['creation_ts']

        formatted_bug['short_desc'] = bug['short_desc']
        formatted_bug['product'] = ''

        if 'component' in bug:
            formatted_bug['component'] = bug['component']
        else:
            formatted_bug['component'] = ''

        if 'version' in bug:
            formatted_bug['version'] = bug['version']
        else:
            formatted_bug['version'] = ''
        
        formatted_bug['bug_status'] = bug['bug_status']
        # formatted_bug['resolution'] = bug['resolution']

        if 'priority' in bug:
            formatted_bug['priority'] = bug['priority']
        else:
            formatted_bug['priority'] = ''

        if 'bug_severity' in bug:
            formatted_bug['bug_severity'] = bug['bug_severity']
        else:
            formatted_bug['bug_severity'] = ''
        
        formatted_bug['description'] = cleantext
        if 'dup_id' in bug:
            real_dup_id = re.findall(r'\d+', str(bug['dup_id']))
            formatted_bug['dup_id'] = real_dup_id[0]

        jsonFile.write(ujson.dumps(formatted_bug))
        jsonFile.write('\n')


def extract_ground_truth(bug_id, comment_dict, cur_address):
    """
    extract dup_id if available
    """

    dup_id = []

    for node in comment_dict['nodes']:
        text = node['body']
        # tmp_dup_list = re.findall(r'dup[a-z]* [a-z]* #\d+', text, flags=re.IGNORECASE)

        # UPD on 23 Dec 2021
        tmp_dup_list = re.findall(r'dup[a-z]* of (\S+)', text, flags=re.IGNORECASE)
    
        if len(tmp_dup_list) > 0:
            for tmp_dup in tmp_dup_list:
                has_https = re.findall(r'https://github.com/(.*)/issues/\d+', tmp_dup)
                
                if len(has_https) > 0:
                    ## different repo
                    if has_https[0].lower() != cur_address:
                        diff_repo.append(bug_id + ' ' + has_https[0])
                        continue
                
                has_num = re.findall(r'\d+', tmp_dup)
                if len(has_num) > 0:
                    dup_id.append(has_num[0])
        else:
            tmp_dup_list = re.findall(r'dup[a-z]* with \S+', text, flags=re.IGNORECASE)
            
            for tmp_dup in tmp_dup_list:
                has_https = re.findall(r'https://github.com/(.*)/issues/\d+', tmp_dup)
                
                if len(has_https) > 0:
                    ## different repo
                    if has_https[0].lower() != cur_address:
                        diff_repo.append(bug_id + ' ' + has_https[0])
                        continue
                
                has_num = re.findall(r'\d+', tmp_dup)
                if len(has_num) > 0:
                    dup_id.append(has_num[0])
                
    if len(dup_id) > 0 and not bug_id in set(dup_id):
        return dup_id
    return ""


def extract_resolution_status():
    status_resolution = list()

    with open('status_resolution.txt', 'a+') as f:
        for line in status_resolution:
            f.write(str(line))
            f.write('\n')


def extract_info_kibana(json_file):
    with codecs.open(json_file, 'r', encoding='utf-8') as f:
        cur_json = ujson.loads(f.read())

    cur_dict = {}
    cur_dict['bug_id'] = os.path.split(json_file)[1].split('.')[0]
    raw_description = cur_json['bodyText']

    # extract the description
    # should we extract the textual part?
    cur_dict['description'] = raw_description

    # 'state': Can be either open, closed, or all.
    cur_dict['bug_status'] = cur_json['state']

    # check the datetime
    creation_date = cur_json['createdAt']
    creation_utc = convert_to_utc(created_date=creation_date)
    cur_dict['creation_ts'] = creation_utc

    if creation_utc < start_date or creation_utc > end_date:
        # not within our time range
        return -1

    closed = cur_json['closed']

    if not closed:
        # open bugs
        return -1

    cur_dict['short_desc'] = cur_json['title']

    ## extract priority
    # total_labels = cur_json['labels']['totalCount']
    # for i in range(total_labels):
    #     cur_label_name = cur_json['labels']['nodes'][i]['name']

    #     if cur_label_name == 'impact:medium':
    #         cur_dict['priority'] = 'medium'
    #     elif cur_label_name == 'impact:critical':
    #         cur_dict['priority'] = 'critical'
    #     elif cur_label_name == 'impact:high':
    #         cur_dict['priority'] = 'high'
    #     elif cur_label_name == 'impact:low':
    #         cur_dict['priority'] = 'low'

    ## extract version
    tmp_list = re.findall(r'Kibana version:\s*\d+.*', cur_json['bodyText'], re.IGNORECASE)
    if len(tmp_list) > 0:
        version_list = re.findall(r'\d+\.*\d*\.*\d*', tmp_list[0], re.IGNORECASE)
        cur_dict['version'] = version_list[0]


    dup_id = extract_ground_truth(cur_dict['bug_id'], cur_json['comments'], kibana_address)
    if len(dup_id) > 0:
        cur_dict['dup_id'] = dup_id

    return cur_dict


def extract_info_vscode(json_file):
    with codecs.open(json_file, 'r', encoding='utf-8') as f:
        cur_json = ujson.loads(f.read())

    cur_dict = {}
    cur_dict['bug_id'] = os.path.split(json_file)[1].split('.')[0]
    cur_dict['description'] = cur_json['bodyText']

    # 'state': Can be either open, closed, or all.
    cur_dict['bug_status'] = cur_json['state']

    # check the datetime
    creation_date = cur_json['createdAt']
    creation_utc = convert_to_utc(created_date=creation_date)
    cur_dict['creation_ts'] = creation_utc

    if creation_utc < start_date or creation_utc > end_date:
        # not within our time range
        return -1

    closed = cur_json['closed']

    if not closed:
        # open bugs
        return -1

    cur_dict['short_desc'] = cur_json['title']

    ## extract priority
    # total_labels = cur_json['labels']['totalCount']
    # for i in range(total_labels):
    #     cur_label_name = cur_json['labels']['nodes'][i]['name']
    #     if cur_label_name == 'important':
    #         cur_dict['priority'] = 'important'

    ## extract version
    tmp_list = re.findall(r'vs\s*code version:\s*\d+.*', cur_json['bodyText'], re.IGNORECASE)
    if len(tmp_list) > 0:
        try:
            version_list = re.findall(r'\d+\.?\d*\.?\d*\.?\d*', tmp_list[0], re.IGNORECASE)
            cur_dict['version'] = version_list[0]
        except IndexError:
            print(json_file)
            sys.exit(0)

    dup_id = extract_ground_truth(cur_dict['bug_id'], cur_json['comments'], vscode_address)
    if len(dup_id) > 0:
        cur_dict['dup_id'] = dup_id

    return cur_dict


def parse_all_jsons():
    dict_list = list()

    for file_name in tqdm(os.listdir(path)):

        full_name = os.path.join(path, file_name)

        # (1) Not a json, remove it
        if not full_name.endswith('json'):
            logging.info('Not JSON')
            os.remove(full_name)
            continue

        if args.project == 'vscode':
            to_add_dict = extract_info_vscode(full_name)
        else:
            to_add_dict = extract_info_kibana(full_name)

        # either invalid or open bug or not within time range
        if to_add_dict == -1:
            continue

        dict_list.append(to_add_dict)

    new_list = sorted(dict_list, key = lambda k: int(k['bug_id']))
    
    with codecs.open(latest_raw_json, 'w', encoding = 'utf-8') as jsonFile:
        for item in new_list:
            jsonFile.write(ujson.dumps(item))
            jsonFile.write('\n')
        
    with open('{}_diff_dup.txt'.format(args.project), 'w') as f:
        for line in diff_repo:
            f.write(line)
            f.write('\n')
    print('has duplicate bug reports from other repos: {}'.format(len(diff_repo)))


def count_duplicate_labels():
    duplicate_labels = []

    for file_name in tqdm(os.listdir(path)):
        full_name = os.path.join(path, file_name)

        with codecs.open(full_name, 'r', encoding='utf-8') as f:
            cur_json = ujson.loads(f.read())

        cur_bug_id = os.path.split(full_name)[1].split('.')[0]

        labels = cur_json['labels']
        for label_node in labels['nodes']:
            cur_label_name = label_node['name']

            if len(re.findall(r'duplicate', cur_label_name)) > 0:
                duplicate_labels.append(cur_bug_id)
                break

    with codecs.open('./{}_duplicate_bugs.txt'.format(args.project), 'a', 'utf-8') as f:
        for item in duplicate_labels:
            f.write(item)
            f.write('\n')
    print('written')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--project', help='project name')

    args = parser.parse_args()
    
    msr_data_path = '/home/zhaozheng/my_projects/TOSEM_DBRD/SABD/dataset/{}/'.format(args.project)

    os.makedirs(msr_data_path, exist_ok=True)
    os.makedirs('/home/zhaozheng/our_data/unused_{}/'.format(args.project), exist_ok=True)

    latest_raw_json = msr_data_path + '{}_raw_latest.json'.format(args.project)
    latest_json = msr_data_path + '{}_latest.json'.format(args.project)

    path = '/home/zhaozheng/our_data/{}'.format(args.project)

    # count_duplicate_labels()
    # if not os.path.exists(latest_raw_json):
    #     logging.info('pasring all the jsons in {}'.format(args.project))
    #     parse_all_jsons()
    parse_all_jsons()
    # extract_resolution_status()
    # if not os.path.exists(latest_json):
    #     logging.info('cleaning the description in {}'.format(args.project))
    #     clean_description(latest_raw_json)
    clean_description(latest_raw_json)