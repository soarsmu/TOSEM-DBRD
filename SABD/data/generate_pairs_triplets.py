"""
### generating training
python data/generate_pairs_triplets.py --bug_data dataset/eclipse/eclipse.json --dataset dataset/eclipse/training_split_eclipse.txt --n 1 --type random

### generating validation
python data/generate_pairs_triplets.py --bug_data dataset/vscode/vscode.json --dataset dataset/vscode/test_vscode.txt --n 1 --type random

"""

import argparse
import logging
import os
import random
random.seed(42)

from itertools import combinations
from time import time

import numpy as np
import sys
sys.path.append('.')


from data.bug_report_database import BugReportDatabase
from data.create_dataset_deshmukh import savePairs, saveTriplets
from data.bug_dataset import BugDataset


class RandomNonDuplicateGenerator(object):

    def __init__(self, bugIds):
        self.bugIds = list(bugIds)
        self.generatedNeg = set()

    @staticmethod
    def randBugId(list):
        return list[random.randint(0, len(list) - 1)]

    def generateNegativeExample(self, n, bugId, masterSet):
        p = 0
        while p < n:
            bug2 = self.randBugId(self.bugIds)

            if bugId == bug2:
                continue

            # Check if bug is not in the same master set
            if bug2 in masterSet:
                continue

            # Check if that a negative example was already generated for that bug
            if bugId > bug2:
                pair = (bugId, bug2)
            else:
                pair = (bugId, bug2)

            if pair in self.generatedNeg:
                continue

            yield bug2
            self.generatedNeg.add(pair)
            p += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--bug_data', required=True, help="")
    parser.add_argument('--dataset', required=True, help="")
    parser.add_argument('--n', required=True, type=int, help="")
    parser.add_argument('--type', required=True, help="")
    parser.add_argument('--aux_file', help="")
    parser.add_argument('--model', help="")

    logging.basicConfig(level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    args = parser.parse_args()
    logger.info(args)

    # if os.path.exists(os.path.join(os.path.split(args.dataset)[0], os.path.split(args.dataset)[1].split('.')[0] + '_pairs_random_1.txt')):
    #     logging.info('the file already exists, will not generate again')
    #     sys.exit()

    bugDataset = BugDataset(args.dataset)
    bugReportDatabase = BugReportDatabase.fromJson(args.bug_data)

    bugIds = bugDataset.bugIds
    duplicateBugs = bugDataset.duplicateIds

    if args.aux_file:
        '''
        In our methodology, we compare new bug with all the previously bugs that are in the database.
        To better generate pairs and triplets, we use the bugs that were reported before the ones 
        from the validation.
        '''
        auxBugDataset = BugDataset(args.aux_file)
        bugsFromMainFile = list(bugIds)
        bugsFromMainFileSet = set(bugIds)

        for auxBugIds in auxBugDataset.bugIds:
            bugIds.append(auxBugIds)

        useAuxFile = True
    else:
        bugsFromMainFile = list(bugIds)
        bugsFromMainFileSet = set(bugIds)
        useAuxFile = False

    # Insert all master ids to the bug id list. The master can be in another file and we need them at least to create a 1 pair
    masterSetById = bugReportDatabase.getMasterSetById(bugIds)

    for masterId in masterSetById.keys():
        bugIds.append(masterId)

    masterIdByBugId = bugReportDatabase.getMasterIdByBugId(bugIds)
    
    # Convert to set to avoid duplicate bug ids
    bugIds = set(bugIds)

    triplets = []
    pairs = []

    if args.type == 'random':
        generator = RandomNonDuplicateGenerator(bugIds)
    else:
        raise Exception('%s is not available option' % args.type)

    total = len(masterSetById)
    current = 0
    last = time()
    emptySet = set()
    listBugIds = list(bugIds)

    logger.info("Generating triplets and pairs")
    for masterId, masterset in masterSetById.items():
        
        for duplicatePair in combinations(masterset, 2):
            if useAuxFile:
                if duplicatePair[0] not in bugsFromMainFileSet and duplicatePair[1] not in bugsFromMainFileSet:
                    # Duplicate pairs that none of the bugs are in the file input were already generated in training dataset
                    continue
                
            for negativeExample in generator.generateNegativeExample(args.n, duplicatePair[0], masterset):
                triplets.append((duplicatePair[0], duplicatePair[1], negativeExample))

        if current != 0 and current % 400 == 0:
            c = time()
            logger.info("Processed %d mastersets of %d. Time: %f" % (current, total, c - last))
            last = time()

        current += 1

    pairs = set()

    for anchor, dup, nondup in triplets:
        pairs.add((anchor, dup, 1))
        pairs.add((anchor, nondup, -1))

    # Check if there are duplicate pairs and triplets
    nPairs = len(pairs)

    if len(set(pairs)) != nPairs:
        print('A duplicate pair was found!')
        sys.exit(-1)

    nTriplets = len(triplets)

    if len(set(triplets)) != nTriplets:
        print('A duplicate triplet was found!')
        sys.exit(-1)


    # Check if the bugs were labeled wrongly
    masterIdByBugId = bugReportDatabase.getMasterIdByBugId()

    for b,p,n in triplets:
        if masterIdByBugId[b] != masterIdByBugId[p]:
            print('Triplets: Positive bug is not correct! (%s,%s)' % (b,p))
            sys.exit(-1)

        if masterIdByBugId[b] == masterIdByBugId[n]:
            print('Triplets: Negative bug is not correct! (%s,%s)' % (b,n))
            sys.exit(-1)

    for b1,b2,l in pairs:
        if l == 1 and masterIdByBugId[b1] != masterIdByBugId[b2]:
            print('Positive bug is not correct! (%s,%s)' % (b1,b2))
            sys.exit(-1)

        if l == -1 and masterIdByBugId[b1] == masterIdByBugId[b2]:
            print('Negative bug is not correct! (%s,%s)' % (b1,b2))
            sys.exit(-1)


    part1, part2 = os.path.splitext(args.dataset)

    name = os.path.splitext(os.path.split(args.model)[1])[0] if args.model else args.type

    if not os.path.exists(os.path.join(os.path.split(args.dataset)[0], os.path.split(args.dataset)[1].split('.')[0] + '_pairs_random_1.txt')):
        logging.info('the pairs file does not exist, generating now')
        savePairs(pairs, "%s_pairs_%s_%d%s" % (part1, name, args.n, part2))
    
    if not os.path.exists(os.path.join(os.path.split(args.dataset)[0], os.path.split(args.dataset)[1].split('.')[0] + '_triplets_random_1.txt')):
        logging.info('the triplets file does not exist, generating now')
        saveTriplets(triplets, "%s_triplets_%s_%d%s" % (part1, name, args.n, part2))

    nDupPairs = np.asarray([l if l == 1 else 0 for b1, b2, l in pairs]).sum()
    logger.info(
        '%d duplicate pairs\t%d non-duplicate pairs\t%d pairs' % (nDupPairs, len(pairs) - nDupPairs, len(pairs)))
    logger.info("Total triplets: %d" % len(triplets))