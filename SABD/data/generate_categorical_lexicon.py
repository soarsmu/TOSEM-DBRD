"""
python data/generate_categorical_lexicon.py --bug_data dataset/eclipse/eclipse.json -o dataset/eclipse/categorical_lexicons.json

"""
import argparse
import json
import logging

import sys
sys.path.append('.')

from data.bug_report_database import BugReportDatabase
from data.preprocessing import TransformLowerCaseFilter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bug_data', required=True, help="")
    parser.add_argument('--output', '-o', required=True, help="")

    logging.basicConfig(level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    args = parser.parse_args()
    logger.info(args)

    bugDataset = BugReportDatabase.fromJson(args.bug_data)

    fields = ['product', 'component', 'version', 'bug_severity', 'priority']
    sets = [set() for _ in fields]

    lowerCaseFilter = TransformLowerCaseFilter()

    for bug in bugDataset.bugList:
        for s,f in zip(sets,fields):
            s.add(lowerCaseFilter.filter(bug[f], None))


    unk = ['UUUKNNN']
    dict = {}

    for idx, (f,s) in enumerate(zip(fields,sets)):
        dict[f] = unk + list(s)

    json.dump(dict, open(args.output, 'w'))