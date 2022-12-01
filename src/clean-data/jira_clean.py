"""
Created on 22 July 2021
Modified in Dec 2021
in Aug 2022
"""

import re
import xml.etree.ElementTree as ET
import logging
import os
import ujson
from bs4 import BeautifulSoup
import codecs
from tqdm import tqdm
from datetime import datetime, timezone
import argparse


# start_date = '2018-01-01 00:00:00 +0000'
# end_date = '2020-12-31 23:59:59 +0000'

start_date = '2012-01-01 00:00:00 +0000'
end_date = '2014-12-31 23:59:59 +0000'

bug_groups = list()

def convert_to_utc(created_date):
    dt = datetime.strptime(created_date, '%a, %d %b %Y %H:%M:%S %z').astimezone(tz = timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M:%S %z')

def is_closed(bug_dict):
    """
    return value:
        1 -> closed
        0 -> open
        -1 -> invalid
    """
    try:
        # closed bugs have resolution {'Pending Closed', 'Implemented', 'Fixed', 'Workaround',
        # 'Not A Problem', 'Delivered', 'Done', 'Cannot Reproduce', 'Not A Bug', 'Later',
        # "Won't Fix", 'Auto Closed', 'Invalid', 'Resolved', 'Incomplete', 
        # 'Information Provided', 'Abandoned', "Won't Do", 'Works for Me', 'Duplicate'

        if bug_dict['status'].lower() in {'resolved', 'closed', 'done'}:
            return 1
        return 0
    except AttributeError:
        # open bugs: without 'resolution' value
        # logging.info('{} is open'.format(bug_dict['bug_id']))
        return 0
    except KeyError:
        # invalid bugs
        if len(bug_dict) == 1:
            return -1
        else:
            logging.info('{} KeyError'.format(bug_dict['bug_id']))
            return 0


def extract_ground_truth(root):
    """
    extract dup_id if available, dup_id is ground_truth
    """
    
    master_ids = set()
    
    for i in range(len(root[0][5])):
        if root[0][5][i].tag == 'issuelinks':
            for j in range(len(root[0][5][i])):
                # issuelinktype: can be many
                # logging.info(root[0][5][i][j][0].tag + ': ' + root[0][5][i][j][0].text)
                for k in range(1, len(root[0][5][i][j])):
                    # issuelink
                    cur_tag = root[0][5][i][j][k].tag
                    cur_attr = root[0][5][i][j][k].attrib
                    # if cur_tag == 'outwardlinks' and cur_attr['description'] == 'duplicates':
                    #     # logging.info(root[0][5][i][j][1][0][0].tag + ': ' + root[0][5][i][j][1][0][0].text)
                    #     # not *text*, text is project id, we need *attrib*, real id
                    #     logging.info(root[0][5][i][j][1][0][0].text)
                    #     return root[0][5][i][j][1][0][0].attrib['id']
                    #     logging.info('\n')
                        
                    ### updated on 4 Aug
                    if cur_attr['description'] == 'is duplicated by' or \
                            cur_attr['description'] == 'duplicates':
                        # logging.info(root[0][5][i][j][1][0][0].tag + ': ' + root[0][5][i][j][1][0][0].text)
                        # not *text*, text is project id, we need *attrib*, real id
                        # logging.info(root[0][5][i][j][1][0][0].text)
                        
                        for p in range(len(root[0][5][i][j][k])):
                            master_id = root[0][5][i][j][k][p][0].attrib['id']
                            master_ids.add(master_id)
                        
    return master_ids
    

def clean_description(file_name, cleaned_json):
    with codecs.open(file_name, 'r', 'utf-8') as f:
        lines = f.readlines()

    jsonFile = codecs.open(cleaned_json, 'w', encoding = 'utf-8')
    bugs_seen = set()

    for line in tqdm(lines):
        bug = ujson.loads(line)
        formatted_bug = {}
        formatted_bug['bug_id'] = bug['bug_id']
        if bug['bug_id'] in bugs_seen:
            continue
        bugs_seen.add(bug['bug_id'])
        formatted_bug['key'] = bug['key']

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
        formatted_bug['resolution'] = bug['resolution']

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
            formatted_bug['dup_id'] = bug['dup_id']

        jsonFile.write(ujson.dumps(formatted_bug))
        jsonFile.write('\n')
    
    
def extract_info_from_xml(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

    except ET.ParseError:
        try:
            # incomplete xml file
            with open(xml_file, 'r') as f:
                xml_string = f.read()

            text = re.sub(u"[\x00-\x08\x0b-\x0c\x0e-\x1f]+" , u"", xml_string)
            root = ET.fromstring(text)
        except ET.ParseError:
            return -1

    cur_dict = {}
    # wrong: project id
    # cur_dict['bug_id'] = os.path.split(xml_file)[1].split('.')[0]

    # root[0][5] = 'item'
    components, versions = set(), set()
    for i in range(len(root[0][5])):
        if root[0][5][i].tag == 'key':
            # correct
            # real id
            cur_dict['bug_id'] = root[0][5][i].attrib['id']
            cur_dict['key'] = root[0][5][i].text

        elif root[0][5][i].tag in ['summary', 'priority', \
            'version', 'component', \
            'created', 'updated', 'status', 'resolution', 'description']:
                # and not root[0][5][i].tag in cur_dict:
            cur_dict[root[0][5][i].tag] = root[0][5][i].text
        # if root[0][5][i].tag == 'component':
        #     components.add(root[0][5][i].text)
            
        # elif root[0][5][i].tag in ['version', 'component']:
        #     components += 
    # return len(components)

    # check the datetime
    creation_date = cur_dict.pop('created')
    creation_utc = convert_to_utc(created_date=creation_date)
    cur_dict['creation_ts'] = creation_utc

    if creation_utc < start_date or creation_utc > end_date:
        return -1

    closed = is_closed(cur_dict)

    if closed == -1:
        return -1
    elif closed == 0:
        return -1

    bug_status = cur_dict.pop('status')
    cur_dict['bug_status'] = bug_status

    short_desc = cur_dict.pop('summary')
    cur_dict['short_desc'] = short_desc
    
    dup_ids = extract_ground_truth(root)
    
    if len(dup_ids) > 0:
        dup_ids.add(cur_dict['bug_id'])
    else:
        return cur_dict
    
    found_set = False
    
    for bug_group_item in bug_groups:
        if len(bug_group_item.intersection(dup_ids)) > 0:
            found_set = True
            
            for item in dup_ids:
                bug_group_item.add(item)

    if not found_set:
        bug_groups.append(dup_ids)
        
    return cur_dict

def parse_all_xmls():
    dict_list = list()
    
    for file_name in tqdm(os.listdir(raw_xml_path)):
        full_name = os.path.join(raw_xml_path, file_name)
        # (1) Not a XML file
        if not full_name.endswith('xml'):
            continue

        to_add_dict = extract_info_from_xml(full_name)
        
        # either invalid or open bug or not within time range
        if to_add_dict == -1:
            os.remove(full_name)
            continue

        dict_list.append(to_add_dict)

    new_list = sorted(dict_list, key = lambda k: int(k['bug_id']))

    with codecs.open(raw_json, 'w', encoding = 'utf-8') as jsonFile:
        for item in new_list:
            jsonFile.write(ujson.dumps(item))
            jsonFile.write('\n')


def set_duplicates(list_set):
    """
    set the earliest one in the group as the master
    """
    bug_masters = dict()
    
    for bug_group in list_set:
        bug_list = sorted(list(bug_group))
        for i in range(1, len(bug_list)):
            if bug_list[i] in bug_masters:
                print('error')
            bug_masters[bug_list[i]] = bug_list[0]
            
    with open(raw_json) as f:
        lines = f.readlines()
    
    updated_lines = list()
    
    for line in lines:
        cur_bug = ujson.loads(line)
        
        if cur_bug['bug_id'] in bug_masters:
            cur_bug['dup_id'] = bug_masters[cur_bug['bug_id']]
            updated_lines.append(cur_bug)
        else:
            updated_lines.append(cur_bug)
        
    with open(latest_json, 'w') as f:
        for line in updated_lines:
            f.write(ujson.dumps(line))
            f.write('\n')
        
def count_compoents():
    component_count, more_components = 0, 0
    zero_components = 0
    for file_name in tqdm(os.listdir(raw_xml_path)):
        full_name = os.path.join(raw_xml_path, file_name)
        num_compoents = extract_info_from_xml(full_name)
        if num_compoents == 1:
            component_count += 1
        elif num_compoents == 0:
            zero_components += 1
        else:
            more_components += 1
    print('0 compoents: {}'.format(zero_components))
    print('1 compoents: {}'.format(component_count))
    print('more than 1 compoents: {}'.format(more_components))
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--project', help='project name', required=True)

    args = parser.parse_args()
    
    data_path = '../../SABD/dataset/{}/'.format(args.project)
    os.makedirs(data_path, exist_ok=True)

    # latest_raw_json = data_path + '{}_raw_latest.json'.format(args.project)
    raw_json = data_path + '{}_raw.json'.format(args.project)
    latest_json = data_path + '{}_latest.json'.format(args.project)
    raw_xml_path = '../../SABD/dataset/jira/{}'.format(args.project)

    parse_all_xmls()
    
    print(len(bug_groups))
    
    for i in range(len(bug_groups)):
        for j in range(i + 1, len(bug_groups)):
            if len(bug_groups[i].intersection(bug_groups[j])) > 0:
                print(bug_groups[i])
                print(bug_groups[j])
                print()
                for item in bug_groups[j]:
                    bug_groups[i].add(item)
                bug_groups[j] = set()
    
    with open(data_path + '{}_bugs.txt'.format(args.project), 'w') as f:
        for bug_group in bug_groups:
            f.write(','.join(sorted(list(bug_group))))
            f.write('\n')
    
    set_duplicates(bug_groups)
    clean_description(latest_json, raw_json)
    # count_compoents()