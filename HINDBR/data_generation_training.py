import sys
sys.path.append('./')
import pandas as pd
from tqdm import tqdm
import ujson
import logging
import argparse
import os

''' Generage model training data for training and test HINDBR'''


def generate_corpus():
    """
    这是提前把信息存下来，summary和一些结构化信息
    不需要description
    """
    
    columns = ["bid", "summary", "pro", "com", "ver","sev", "pri", "sts"]

    df = pd.DataFrame(columns=columns)
    df[['bid']] = df[['bid']].astype('int')
    df[['summary']] = df[['summary']].astype('object')
    df[['pro']] = df[['pro']].astype('object')
    df[['com']] = df[['com']].astype('object')
    df[['ver']] = df[['ver']].astype('object')
    df[['sev']] = df[['sev']].astype('object')
    df[['pri']] = df[['pri']].astype('object')
    df[['sts']] = df[['sts']].astype('object')

    with open(JSON_FILE_PATH) as f:
        lines = f.readlines()

    for i in tqdm(range(len(lines))):
        line = lines[i]
        cur_dict = ujson.loads(line)
        bug_id = cur_dict['bug_id']
        
        short_desc = cur_dict['short_desc']
        if len(short_desc) != 0:
            # Output the raw summary text
            bug_summary = short_desc
        else:
            bug_summary = ''

        # Product
        bug_product = cur_dict['product']
        if len(bug_product) != 0:
            bug_product = 'PRO_' + bug_product
        else:
            bug_product = ''

        # Component
        bug_component = cur_dict['component']
        if len(bug_component) != 0:
            bug_component = 'COM_' + bug_component
        else:
            bug_component = ''

        # Version
        bug_version = cur_dict['version']
        if len(bug_version) != 0:
            bug_version = 'VER_' + bug_version
        else:
            bug_version = ''

        # Severity
        bug_severity = cur_dict['bug_severity']
        if len(bug_severity) != 0:
            bug_severity = 'SEV_' + bug_severity
        else:
            bug_severity = ''

        # Priority
        bug_priority = cur_dict['priority']
        if len(bug_priority) != 0:
            bug_priority = 'PRI_' + bug_priority
        else:
            bug_priority = ''

        # Bug Status
        bug_status = cur_dict['bug_status']
        if len(bug_status) != 0:
            bug_status = 'STS_' + bug_status
        else:
            bug_status = ''

        
        info_dict = {
            'bid': bug_id,
            'summary': bug_summary,
            'pro': bug_product,
            'com': bug_component,
            'ver': bug_version,
            'sev': bug_severity,
            'pri': bug_priority,
            'sts': bug_status
        }

        df.loc[i] = info_dict
    
    os.makedirs('data/model_training', exist_ok=True)
    df.to_pickle('data/model_training/{}_corpus.pkl'.format(PROJECT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter project name.')
    parser.add_argument('--project', help='project name', required=True)
    
    args = parser.parse_args()
    PROJECT = args.project

    JSON_FILE_PATH = '../SABD/dataset/{}/{}_soft_clean.json'.format(PROJECT, PROJECT)

    TRAIN_CSV = 'data/model_training/' + PROJECT + '_train.csv'
    VALID_CSV = 'data/model_training/' + PROJECT + '_valid.csv'
    TEST_CSV = 'data/model_test/' + PROJECT + '_test.csv'

    generate_corpus()