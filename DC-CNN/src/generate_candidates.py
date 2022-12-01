"""
Created on 3 Sep 2022
"""

import sys
sys.path.append('./')
import argparse
from modules import BugReportDatabase, BugDataset, SunRanking
from pathlib import Path
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter project name.')
    parser.add_argument('--project', help='project name', required=True)
    
    args = parser.parse_args()
    
    PROJECT = args.project
    data_path = Path('../data/')
    sabd_data_path = '../../SABD/dataset/{}/'.format(PROJECT)
    bug_report_database = BugReportDatabase.fromJson(sabd_data_path + '{}.json'.format(PROJECT))
    recallRateDataset = BugDataset(sabd_data_path + 'test_{}.txt'.format(PROJECT))
    duplicateBugs = recallRateDataset.duplicateIds
    rankingClass = SunRanking(bug_report_database, recallRateDataset, 365)
    positions = []

    with open('../{}_candidates.txt'.format(PROJECT), 'w') as f:
        for i, duplicateBugId in enumerate(tqdm(duplicateBugs)):
            candidates = rankingClass.getCandidateList(duplicateBugId)
            f.write(' '.join([str(c) for c in candidates]))
            f.write('\n')