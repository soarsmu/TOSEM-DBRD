"""
Created by author
on 13 Jun 2021

some functions are borrowed from msr20
"""


import random
random.seed(941207)

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import re

import codecs
import ujson


def preprocess_bugs():
    with codecs.open(latest_soft_cleaned_json, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        item = ujson.loads(line)

        bug_id = int(item['bug_id'])

        try:
            product_info = 'product ' + item['product']
            component_info = ' component ' + item['component']
            summary_info = ' summary ' + item['short_desc']
            description_info = ' description ' + item['description']
            to_write_str = ''.join([product_info, component_info, summary_info, description_info])
            to_write_str = to_write_str.lower()
            
            cleaned = re.sub('\s+', ' ', to_write_str)
            with open(os.path.join(project_path, str(bug_id) + '.txt'), 'w') as f:
                f.write('%s' % cleaned)

        except KeyError:
            # without description
            product_info = 'product ' + item['product']
            component_info = ' component ' + item['component']
            summary_info = ' summary ' + item['short_desc']
            description_info = ' description '
            to_write_str = ''.join([product_info, component_info, summary_info, description_info])
            
            to_write_str = to_write_str.lower()
            cleaned = re.sub('\s+', ' ', to_write_str)
            with open(os.path.join(project_path, str(bug_id) + '.txt'), 'w') as f:
                f.write('%s' % cleaned)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--project', help='project name')

    args = parser.parse_args()
    
    project_path = Path('../data/preprocess/{}/'.format(args.project))
    project_path.mkdir(parents=True, exist_ok=True)

    sabd_data_path = '../../SABD/dataset/{}/'.format(args.project)
    latest_soft_cleaned_json = sabd_data_path + '{}_soft_clean.json'.format(args.project)

    training_file = sabd_data_path + 'training_{}.txt'.format(args.project)
    
    preprocess_bugs()