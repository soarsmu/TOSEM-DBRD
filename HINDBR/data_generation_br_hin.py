"""
Modified from data_generation_br_hin.py
e.g., python data_generation_br_hin.py --project kibana
on 3 Aug 2021
"""

import logging
import json
from modules import *
import ujson
import codecs
import argparse
from tqdm import tqdm
import os, sys
sys.path.append('./')

'''Generate bug report heterogeneous information network '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter project name.')
    parser.add_argument('--project', help='project name', required=True)

    args = parser.parse_args()
    PROJECT = args.project
    
    os.makedirs('data/bug_report_hin', exist_ok=True)
    
    HINFILENAME = 'data/bug_report_hin/' + PROJECT + '.hin'

    JSON_FILE_PATH = '../SABD/dataset/{}/{}_soft_clean.json'.format(PROJECT, PROJECT)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('Generate bug report hin for the ' + PROJECT + ' project!')
    logging.info('loading the json file...')


    with open(HINFILENAME, 'w') as f:
        f.write('#source_node' + '\t' + 'source_class' + '\t' + \
        'dest_node' + '\t' + 'dest_class' + '\t' + 'edge_class' + '\n')

        with codecs.open(JSON_FILE_PATH, 'r', 'utf-8') as jsonHandler:
            lines = jsonHandler.readlines()

        for i, line in tqdm(zip(range(len(lines)), lines), total=len(lines)):
            if i % 1000 == 0:
                logging.info("{0} bugs have been processed!".format(i))

            # node generation
            node_result = nodeGenerationFromJson(ujson.loads(line))

            # edge generation
            edge_result = edgeGeneration(node_result[0], node_result[1], 'default')
            for edge in edge_result:
                f.write(edge)

    # store and output hin nodes' dictionary
    node_dict = node_result[1]
    js = json.dumps(node_dict)
    
    os.makedirs('data/hin_node_dict', exist_ok=True)
    with open('data/hin_node_dict/' + PROJECT + '_node.dict','w') as f:
        f.write(js)

    # store and output hin nodes' classes
    with open('data/hin_node_dict/' + PROJECT + '_node_class.txt', 'w') as f:
        f.write('node_id' + '\t' + 'node class (separated by tab)' + '\n')
        for node in node_dict:
            node_id = node_dict[node][0]
            node_class = node_dict[node][1]
            f.write(str(node_id) + '\t' + node_class + '\n')

    logging.info('HIN generation done!')