"""
We investigate the cause of failed duplicate detection.
"""

from utils import get_logger
import logging
import re
import os
import glob
from tqdm import tqdm
import random
import json
random.seed(42)

def get_positions(result_file, failure_file):
    with open(result_file, 'r') as f:
        content = f.read()
        
    index_pos_list = list()
    pos_list = re.findall('INFO: \[(\d+.*)\]', content)[0].split(',')
    for index, item in zip(range(len(pos_list)), pos_list):
        if item.strip() != 'inf' and int(item.strip()) > 10:
            index_pos_list.append((index, int(item.strip())))
    
    with open(failure_file, 'w') as f:
        for index, pos in index_pos_list:
            f.write('{}, {}\n'.format(index, pos))
            
    print('number of failures: {}'.format(len(index_pos_list)))
    print('percentage is: {}'.format(len(index_pos_list) / len(pos_list)))

def get_indexes(failure_file):
    indexes = list()
    with open(failure_file) as f:
        lines = f.readlines()
    for line in lines:
        index = line.split(',')[0]
        indexes.append(index)
    return indexes
    
def get_results(project, approach, folder):
    result_files = glob.glob(folder + '/{}_{}_*.log'.format(approach, project))
    result_files.sort()
    return result_files
    
def get_results_rep(project, folder):
    result_files = glob.glob(folder + '/recommend_ranknet_*_*_*_{}-*_*-*'.format(project))
    real_results = list()
    for file in result_files:
        if 'initial' in file or 'old' in file:
            continue
        real_results.append(file)
    real_results.sort()
    return real_results

def get_failed_data(project, approach, folder):
    """
    generate a list of failed data in all the five runs
    """
    
    logging.info('Getting failed data for {} {}...'.format(project, approach))
    result_files = get_results(project, approach, folder)
    print('number of files: {}'.format(len(result_files)))
    result1 = result_files[0]
    result2 = result_files[1]
    result3 = result_files[2]
    result4 = result_files[3]
    result5 = result_files[4]
    # result1 = '../SABD/result_log/RQ1-Age-Bugzilla/pairs_mozilla_2022-08-17-14:46:32.log'
    # result2 = '../SABD/result_log/RQ1-Age-Bugzilla/pairs_mozilla_2022-08-17-17:11:11.log'
    # result3 = '../SABD/result_log/RQ1-Age-Bugzilla/pairs_mozilla_2022-08-17-19:41:49.log'
    # result4 = '../SABD/result_log/RQ1-Age-Bugzilla/pairs_mozilla_2022-08-17-22:43:23.log'
    # result5 = '../SABD/result_log/RQ1-Age-Bugzilla/pairs_mozilla_2022-08-18-01:11:54.log'
    
    failure1 = 'failed/{}_{}-1.txt'.format(project, approach)
    failure2 = 'failed/{}_{}-2.txt'.format(project, approach)
    failure3 = 'failed/{}_{}-3.txt'.format(project, approach)
    failure4 = 'failed/{}_{}-4.txt'.format(project, approach)
    failure5 = 'failed/{}_{}-5.txt'.format(project, approach)
    
    get_positions(result1, failure1)
    get_positions(result2, failure2)
    get_positions(result3, failure3)
    get_positions(result4, failure4)
    get_positions(result5, failure5)
    
    index_list_1 = get_indexes(failure1)
    index_list_2 = get_indexes(failure2)
    index_list_3 = get_indexes(failure3)
    index_list_4 = get_indexes(failure4)
    index_list_5 = get_indexes(failure5)
    
    intersected_set = set(index_list_1) & set(index_list_2) \
        & set(index_list_3) & set(index_list_4) & set(index_list_5)
    
    test_file = '../SABD/dataset/{}/test_{}.txt'.format(project, project)
    with open(test_file) as f:
        lines = f.readlines()
    intersected_list = list()
    for index, bug_id in zip(range(len(lines[2].split())), lines[2].split()):
        if str(index) in intersected_set:            
            intersected_list.append('{}, {}'.format(index, bug_id))
            
    with open('failed/{}_{}.txt'.format(project, approach), 'w') as f:
        for index in intersected_list:
            f.write(index)
            f.write('\n')
    
def get_overlapp(f1, f2, f3, project):
    set1, set2, set3 = set(), set(), set()
    with open(f1) as f:
        lines = f.readlines()
    for line in lines:
        set1.add(line.strip())
        
    with open(f2) as f:
        lines = f.readlines()
    for line in lines:
        set2.add(line.strip())
        
    with open(f3) as f:
        lines = f.readlines()
    for line in lines:
        set3.add(line.strip())
    overlapped = set1 & set2 & set3
    with open('./failed/overlapped_{}.txt'.format(project), 'w') as f:
        for item in overlapped:
            f.write(item)
            f.write('\n')
            
            
def sample_data(failure_file, project):
    """
    sample 50 BRs from each project by each approach
    """
    with open(failure_file) as f:
        lines = f.readlines()
    sampled_lines = random.sample(lines, 50)

    json_data = '../SABD/dataset/{}/{}.json'.format(project, project)

    with open(json_data) as f:
        lines = f.readlines()
    bug_dup = dict()
    for line in lines:
        bug = json.loads(line)
        bug_id = bug['bug_id']
        dup_id = bug['dup_id']
        bug_dup[bug_id] = dup_id

    fully_sampled_lines = list()
    for line in sampled_lines:
        bug_id = line.strip().split(',')[1].strip()
        dup_id = bug_dup[bug_id]
        fully_sampled_lines.append('{}, {}'.format(bug_id, dup_id))

    with open('failed/{}_sample_overlapped.txt'.format(failure_file.split('/')[-1].split('.')[0]), 'w') as f:
        for line in fully_sampled_lines:
            f.write(line)
            f.write('\n')
    

def extract_rep_position(rep_result, failure_file):
    with open(rep_result) as f:
        lines = f.readlines()

    failed = list()
    for l in tqdm(range(len(lines))):
        line = lines[l]

        iteration = re.findall(r'Iteration \d', line)
        if len(iteration) > 0:
            
            for i in range(l + 1, len(lines), 22):
                if lines[i].strip() == '':
                    break
                
                dup_id = re.findall(r'Retrieving for duplicate report (\d+)', lines[i])[0]
                
                found = False
                for j in range(i + 1, i + 11):
                    if lines[j].strip()[-1] == '+':
                        found = True
                        break
                    
                if not found:
                    failed.append('{}, {}'.format((i - l) // 22, dup_id))
            break
                    
    with open(failure_file, 'w') as f:
        for index_pos in failed:
            f.write('{}\n'.format(index_pos))


def get_failed_rep_data(project):
    result_files = get_results_rep(project, '../REP')
    result1 = result_files[0]
    result2 = result_files[1]
    result3 = result_files[2]
    result4 = result_files[3]
    result5 = result_files[4]
    
    failure1 = 'failed/rep_{}-1.txt'.format(project)
    failure2 = 'failed/rep_{}-2.txt'.format(project)
    failure3 = 'failed/rep_{}-3.txt'.format(project)
    failure4 = 'failed/rep_{}-4.txt'.format(project)
    failure5 = 'failed/rep_{}-5.txt'.format(project)
    
    extract_rep_position(result1, failure1)
    extract_rep_position(result2, failure2)
    extract_rep_position(result3, failure3)
    extract_rep_position(result4, failure4)
    extract_rep_position(result5, failure5)
    
    index_list_1 = get_indexes(failure1)
    index_list_2 = get_indexes(failure2)
    index_list_3 = get_indexes(failure3)
    index_list_4 = get_indexes(failure4)
    index_list_5 = get_indexes(failure5)
    
    intersected_set = set(index_list_1) & set(index_list_2) \
        & set(index_list_3) & set(index_list_4) & set(index_list_5)
    
    test_file = '../SABD/dataset/{}/test_{}.txt'.format(project, project)
    with open(test_file) as f:
        lines = f.readlines()
    intersected_list = list()
    for index, bug_id in zip(range(len(lines[2].split())), lines[2].split()):
        if str(index) in intersected_set:            
            intersected_list.append('{}, {}'.format(index, bug_id))
            
    with open('failed/rep_{}.txt'.format(project), 'w') as f:
        for index in intersected_list:
            f.write(index)
            f.write('\n')
    
if __name__ == '__main__':
    # logger = get_logger('../log/cause_analysis.log')
    os.makedirs('failed', exist_ok=True)
    # get_failed_data('hadoop', 'pairs', '../SABD/result_log/discussion-state-jira')
    # get_failed_data('hadoop', 'sabd', '../SABD/result_log/discussion-state-jira')
    # get_failed_rep_data('hadoop')
    # get_overlapp('failed/hadoop_pairs.txt', 'failed/hadoop_sabd.txt', 'failed/rep_hadoop.txt', 'hadoop')
    # sample_data('failed/overlapped_hadoop.txt')
    get_failed_data('vscode', 'pairs', '../SABD/result_log/RQ1-ITS')
    get_failed_data('vscode', 'sabd', '../SABD/result_log/RQ1-ITS')
    get_failed_rep_data('vscode')
    get_overlapp('failed/vscode_pairs.txt', 'failed/vscode_sabd.txt', 'failed/rep_vscode.txt', 'vscode')
    sample_data('failed/overlapped_vscode.txt', 'vscode')
