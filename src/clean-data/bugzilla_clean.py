"""
Created on 8th June 2021
modified on 7 Dec 2021
This file contains all the essential steps for cleaning the data
"""


import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import logging
import re
from datetime import datetime, timezone
import ujson
import codecs

import argparse


logger = logging.getLogger()
logger.setLevel(logging.INFO)


"""
examples in open_office data
{
    "bug_id":"13",
    "product":"Calc",
    "description":"I need to see if this works, sorry.",
    "bug_severity":"trivial",
    "dup_id":[],
    "short_desc":"Test bug: Cell color is wrong",
    "priority":"P4",
    "version":"605",
    "component":"code",
    "bug_status":"CLOSED",
    "creation_ts":"2000-10-16 18:33:00 +0000",
    "resolution":"NOT_AN_ISSUE"
}
"""

parsed_errors = list()
open_bugs = list()

# convert datetime to UTC first
def convert_to_utc(created_date):
    dt = datetime.strptime(created_date, '%Y-%m-%d %H:%M:%S %z').astimezone(tz = timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M:%S %z')


def parse_all_xmls():
    jsonFile = codecs.open(latest_json, 'w', encoding = 'utf-8')

    dict_list = list()

    for file_name in tqdm(os.listdir(path)):
        full_name = os.path.join(path, file_name)
        # (1) Not a XML, remove it
        if not full_name.endswith('xml'):
            logging.info('No xml')
            os.remove(full_name)
            continue

        to_add_dict = parse_xml(full_name)

        # either invalid or open bug or not within time range
        if to_add_dict == -1:
            continue

        dict_list.append(to_add_dict)

    new_list = sorted(dict_list, key = lambda k: int(k['bug_id']))

    for item in new_list:
        jsonFile.write(ujson.dumps(item))
        jsonFile.write('\n')

def is_closed(bug_dict):
    """
    return value:
        1 -> closed
        0 -> open
        -1 -> invalid
    """

    try:
        if bug_dict['bug_status'].lower() in {'resolved', 'verified', 'closed'} or \
            bug_dict['resolution'].lower() in {
                'fixed', 'invalid', 'wontfix', 'duplicate', 'worksforme', 'incomplete', 'moved'
            }:
            return 1
        return 0
    except AttributeError:
        # open bugs: without 'resolution' value
        # logging.info('{} is open'.format(bug_dict['bug_id']))
        return 0
    except KeyError:
        # invalid bugs
        if len(bug_dict) == 1:
            # logging.info('{} KeyError'.format(bug_dict['bug_id']))
            return -1
        else:
            logging.info('{} KeyError'.format(bug_dict['bug_id']))
            return 0


def extract_bug_report_information(xmlfile):
    with open(xmlfile, 'r') as f:
        contents = f.read()
        # bug id
        bugid = int(re.findall('<bug_id>(.*?)</bug_id>', contents)[0])
        
        # Summary
        short_desc = re.findall('<short_desc>(.*?)</short_desc>', contents)
        if len(short_desc) != 0:
            # Output the raw summary text
            bug_summary = short_desc[0]
        else:
            bug_summary = ''

        # Description
        long_desc = re.findall('<thetext>(.*?)</thetext>', contents, re.DOTALL)
        if len(long_desc) != 0:
            bug_description = long_desc[0]
        else:
            bug_description = ''

        # Product
        bug_product = re.findall('<product>(.*)</product>', contents)
        if len(bug_product) != 0:
            bug_product = bug_product[0]
        else:
            bug_product = ''

        # Component
        bug_component = re.findall('<component>(.*)</component>',contents)
        if len(bug_component) != 0:
            bug_component = bug_component[0]
        else:
            bug_component = ''

        # Version
        if 'linux' in xmlfile:
            bug_version = re.findall('<cf_kernel_version>(.*)</cf_kernel_version>',contents)
        else:
            bug_version = re.findall('<version>(.*)</version>',contents)
            
        if len(bug_version) != 0:
            bug_version = bug_version[0]
        else:
            bug_version = ''

        # Severity
        bug_severity = re.findall('<bug_severity>(.*)</bug_severity>', contents)
        if len(bug_severity) != 0:
            bug_severity = bug_severity[0]
        else:
            bug_severity = ''
            
        # Priority
        bug_priority = re.findall('<priority>(.*)</priority>',contents)
        if len(bug_priority) != 0:
            bug_priority = bug_priority[0]
        else:
            bug_priority = ''

        # Bug Status
        bug_status = re.findall('<bug_status>(.*)</bug_status>',contents)
        if len(bug_status) != 0:
            bug_status = bug_status[0]
        else:
            bug_status = ''


        # Creation time
        bug_reported_time = re.findall('<creation_ts>(.*)</creation_ts>', contents)[0]
        bug_reported_time = convert_to_utc(bug_reported_time)
        
        # get resolution status
        resolution = re.findall('<resolution>(.*?)</resolution>', contents)
        if len(resolution) != 0:
            resolution = resolution[0]
        else:
            resolution = ''
            
        # get dupids if the resolution is duplicate
        if resolution.lower() == 'duplicate':
            dupid = int(re.findall('<dup_id>(.*?)</dup_id>', contents)[0])
        else:
            dupid = []
    
    # open bug reports
    if is_closed({'bug_status': bug_status}) == -1:
        return {}
    
    # out the time scope
    if bug_reported_time < start_date or bug_reported_time > end_date:
        return {}
    
    return {
        'bug_id': bugid,
        'product': bug_product,
        'description': bug_description,
        'bug_severity': bug_severity, 
        'dup_id': dupid,
        'short_desc': bug_summary,
        'priority': bug_priority, 
        'version': bug_version, 
        'component': bug_component, 
        'bug_status': bug_status, 
        'creation_ts': bug_reported_time,
        'resolution': resolution
    }


def extract_all_xmls():
    with codecs.open(latest_json_dec, 'w', encoding = 'utf-8') as jsonFile:
        dict_list = list()

        for file_name in tqdm(os.listdir(path)):
            full_name = os.path.join(path, file_name)
            # (1) Not a XML, remove it
            if not full_name.endswith('xml'):
                logging.info('No xml')
                os.remove(full_name)
                continue

            to_add_dict = extract_bug_report_information(full_name)

            # either invalid or open bug or not within time range
            if to_add_dict == -1:
                continue

            dict_list.append(to_add_dict)

        new_list = sorted(dict_list, key = lambda k: int(k['bug_id']))

        for item in new_list:
            jsonFile.write(ujson.dumps(item))
            jsonFile.write('\n')


def parse_xml(xml_file):
    # create element tree object
    try:
        tree = ET.parse(xml_file)
        # get root element
        root = tree.getroot()

    except ET.ParseError:
        try:
            # incomplete xml file
            with open(xml_file, 'r') as f:
                xml_string = f.read()

            text = re.sub(u"[\x00-\x08\x0b-\x0c\x0e-\x1f]+" , u"", xml_string)
            root = ET.fromstring(text)
        except ET.ParseError:
            # not a xml file at all, maybe a html
            return -1


    cur_dict = {}

    reported_author = ''

    for i in range(len(root[0])):

        if root[0][i].tag == 'reporter':
            reported_author = root[0][i].text

        if root[0][i].tag in ["bug_id", "product", "bug_severity", "dup_id", "short_desc", \
            "priority", "version", "component", "bug_status", "creation_ts", "resolution"]:
            cur_dict[root[0][i].tag] = root[0][i].text
        
        # only select the first long_desc as description
        # and make sure the author and create datetime is the same
        #  as the bug reported time
        if root[0][i].tag == 'long_desc' and 'description' not in cur_dict:
            for j in range(len(root[0][i])):
                if root[0][i][j].tag == 'who':
                    author = root[0][i][j].text

                    if author != reported_author:
                        cur_dict['description'] = ''
                        logging.info('name different failed on {}'.format(xml_file))
                        logging.info('author: {}'.format(author))
                        break
                
                if root[0][i][j].tag == 'bug_when':
                    if cur_dict['creation_ts'] != root[0][i][j].text:
                        cur_dict['description'] = ''
                        logging.info('creation time different failed on {}'.format(xml_file))
                        logging.info('time: {}'.format(root[0][i][j].text))
                        break

                if root[0][i][j].tag == 'thetext':
                    if root[0][i][j].text == None:
                        cur_dict['description'] = ''
                    else:
                        cur_dict['description'] = root[0][i][j].text
                    break

    # check the datetime
    utc_creation_time = convert_to_utc(cur_dict['creation_ts'])
    cur_dict['creation_ts'] = utc_creation_time

    if utc_creation_time < start_date or utc_creation_time > end_date:
        name_tail = os.path.split(xml_file)[1]

        os.makedirs('/home/zhaozheng/our_data/unused_{}/'.format(args.project), exist_ok=True)
        os.rename(xml_file, '/home/zhaozheng/our_data/unused_{}/'.format(args.project) + name_tail)
        return -1

    closed = is_closed(cur_dict)

    if closed == -1:
        # move invalid bugs to invalid_bug folder
        name_tail = os.path.split(xml_file)[1]
        
        os.makedirs('../../our_data/invalid_{}/'.format(args.project), exist_ok=True)
        os.rename(xml_file, '../../our_data/invalid_{}/'.format(args.project) + name_tail)
        return -1
    elif closed == 0:
        # move open bugs to open_bug folder
        name_tail = os.path.split(xml_file)[1]

        os.makedirs('../../our_data/open_bugs_{}/'.format(args.project), exist_ok=True)
        os.rename(xml_file, '../../our_data/open_bugs_{}/'.format(args.project) + name_tail)
        return -1

    return cur_dict


if __name__ == "__main__":
    # parse xml file
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--project', help='project name')

    args = parser.parse_args()
    file_handler = logging.FileHandler('../SABD/log/clean_{}_data.log'.format(args.project))
    
    latest_json = '../SABD/dataset/{}/{}_dec.json'.format(args.project, args.project)
    
    latest_json_dec = '../SABD/dataset/{}/{}_latest_dec.json'.format(args.project, args.project)

    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s', datefmt = '%F %A %T'))
    logger.addHandler(file_handler)

    start_date = '2018-01-01 00:00:00 +0000'
    end_date = '2020-12-31 23:59:59 +0000'


    path = '../../our_data/{}/'.format(args.project)

    parse_all_xmls()
    # extract_all_xmls()