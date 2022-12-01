# Spliting the pairs in training and test dataset
import argparse
import logging

import ujson
from sklearn.model_selection import train_test_split
import itertools
import random
import os

import pymongo
import numpy as np

"""
Create dataset similar to Deshmukh et al, 2018 
"""


def createNonDuplicateBugGenerator(duplicatePairs, masterSet, proportion, pairsDict):
    # Randomly generate X times more non-duplicate pairs than duplicate
    logger = logging.getLogger()
    random.shuffle(masterSet)

    totalNonDuplicatePairs = 0
    proportion = int(proportion)

    for idx, pair in enumerate(duplicatePairs):
        if idx != 0 and idx % 100 == 0:
            random.seed()

        bug = pair[0]
        duplicateBug = pair[1]

        for i in range(proportion):
            while True:
                idx = random.randint(0, len(masterSet) - 1)
                nonDuplicateBug = int(masterSet[idx])

                if nonDuplicateBug != bug:
                    key = createKey(bug, nonDuplicateBug)

                    if key not in pairsDict:
                        pairsDict[key] = True
                        yield bug, duplicateBug, nonDuplicateBug

                        totalNonDuplicatePairs += 1
                        break

                    logger.info("Repeated pair %s" % key)

    logger.info("Number of non duplicate sampled: %d" % totalNonDuplicatePairs)


def generateNonDuplicatePairs(duplicatePairs, masterSet, proportion, pairsDict):
    return [(bug, nonDuplicateBug, -1) for bug, duplicateBug, nonDuplicateBug in
            createNonDuplicateBugGenerator(duplicatePairs, masterSet, proportion, pairsDict)]


def generateTriplets(duplicatePairs, masterSet, proportion, pairsDict):
    return [(bug, duplicateBug, nonDuplicateBug) for bug, duplicateBug, nonDuplicateBug in
            createNonDuplicateBugGenerator(duplicatePairs, masterSet, proportion, pairsDict)]


def getAllDuplicatePairs(database, pairsDict, enableDuplicatePairs=False):
    logger = logging.getLogger()
    client = pymongo.MongoClient()
    database = client[database]
    collection = database['pairs']

    logger.info("Gathering positive examples")
    duplicatePairs = []

    for pair in collection.find({}):
        if pair["dec"] == 1:
            bug1 = pair['bug1']
            bug2 = pair['bug2']
            key = createKey(bug1, bug2)

            if not enableDuplicatePairs and key in pairsDict:
                continue

            duplicatePairs.append((bug1, bug2, 1))
            pairsDict[key] = True

    nDuplicatePairs = len(duplicatePairs)
    logger.info("Number of duplicate pairs: % d" % nDuplicatePairs)

    return duplicatePairs


def getMasterSetAndDuplicateByMaster(database, collection):
    logger = logging.getLogger()
    client = pymongo.MongoClient()

    database = client[database]
    collection = database[collection]

    nbDuplicateRecords = 0
    bugAlreadySeen = {}
    cursor = collection.find({})

    masterByBugId = {}
    bugPerId = {}

    # List with master bugs (bugs that has a empty dup_id value)
    masterIds = set()

    for bug in cursor:
        bugId = bug["bug_id"]
        dupId = bug['dup_id']

        bugPerId[bugId] = bug

        # there are bugs that were inserted twice in the database
        if bugAlreadySeen.get(bugId, False):
            nbDuplicateRecords += 1
            continue
        else:
            bugAlreadySeen[bugId] = True

        if len(dupId) != 0:
            # Create tree of dependence
            masterByBugId[bugId] = dupId
        else:
            masterIds.add(bugId)

    mastersetByMasterId = {}
    nMasterNotFound = 0

    for bugId, masterId in masterByBugId.items():
        # if A is duplicate of B and B is duplicate of C, so A is duplicate of C
        nextMasterId = masterByBugId.get(masterId, None)

        while nextMasterId is not None:
            masterId = nextMasterId
            nextMasterId = masterByBugId.get(masterId, None)
        """
        All bugs with OPEN status were deleted from the dataset, thus some masters cannot exist.
        Instead of Lazar et.al. 2014, we ignore all duplicate bugs that the master is not in the dataset
        """
        if not bugAlreadySeen.get(masterId, False):
            nMasterNotFound += 1
            continue

        masterByBugId[bugId] = masterId

        bug = bugPerId[bugId]
        bug['dup_id'] = masterId

        # Insert bug in the master
        dups = mastersetByMasterId.get(masterId, list())
        dups.append(bugId)

        if len(dups) == 1:
            mastersetByMasterId[masterId] = dups

    logger.info("Number of report that was thrown way because the master was not found in the dataset: %d" % nMasterNotFound)
    logger.info("Number of duplicate records: %d" % nbDuplicateRecords)

    return bugPerId, masterIds, masterByBugId, mastersetByMasterId


def generateAllDuplicatePairs(database, duplicateByMasterId, pairsDict):
    logger = logging.getLogger()
    client = pymongo.MongoClient()
    database = client[database]
    pairsCol = database["pairs"]
    duplicatePairs = []

    # Generate all possible pairs of duplicate bugs, not only the pairs containing the master bug.
    for masterId, dups in duplicateByMasterId.items():
        for bugId1, bugId2 in itertools.combinations(set([masterId] + dups), 2):
            pair = {
                "bug1": bugId1,
                "bug2": bugId2,
                "dec": 1
            }

            duplicatePairs.append(pair)

    nbDuplicatePairs = len(duplicatePairs)

    logger.info("Number of duplicate pairs: % d" % nbDuplicatePairs)
    pairsCol.insert(duplicatePairs)

    for i, pair in enumerate(duplicatePairs):
        duplicatePairs[i] = (pair['bug1'], pair['bug2'], pair['dec'])
        pairsDict[createKey(pair['bug1'], pair['bug2'])] = True

    return duplicatePairs


def tripletToPairs(triples, addPositiveDuplicate=False):
    pairs = []
    keyAlreadySeen = {}

    for (bug, duplicateBug, nonDuplicateBug) in triples:
        duplicateBugKey = createKey(bug, duplicateBug)

        if addPositiveDuplicate or duplicateBugKey not in keyAlreadySeen:
            pairs.append((bug, duplicateBug, 1))
            keyAlreadySeen[duplicateBugKey] = True

        nonDuplicateBugKey = createKey(bug, nonDuplicateBug)

        if nonDuplicateBugKey in keyAlreadySeen:
            raise Exception("We found %s twice." % nonDuplicateBugKey)

        keyAlreadySeen[nonDuplicateBugKey] = True
        pairs.append((bug, nonDuplicateBug, -1))

    return pairs


def savePairs(pairs, fileName):
    logger = logging.getLogger()
    f = open(fileName, "w")
    n = 0

    for (bug1, bug2, label) in pairs:
        f.write("%s,%s,%d\n" % (bug1, bug2, label))
        n += 1

    logger.info("Amount of pairs saved in %s: %d" % (fileName, n))


def saveTriplets(triplets, fileName):
    logger = logging.getLogger()
    f = open(fileName, "w")
    n = 0

    for (bug, duplicateBug, nonDuplicateBug) in triplets:
        f.write("%s,%s,%s\n" % (bug, duplicateBug, nonDuplicateBug))
        n += 1

    logger.info("Amount of triplets saved in %s: %d" % (fileName, n))


def createKey(bug1, bug2):
    return '%s|%s' % tuple(sorted([int(bug1), int(bug2)]))


def hasDuplicatePairs(training, validation, test, ignorePositive=False):
    logger = logging.getLogger()
    d = {}
    totalIgnored = 0

    for n, v in [('tr', training), ('va', validation), ('te', test)]:
        for idx, (bug1, bug2, label) in enumerate(v):
            key = createKey(bug1, bug2)

            if key in d:
                if not ignorePositive or label != 1:
                    s, i = d[key]
                    logger.info("%s (%s,%s,%s) %s %s" % (n, bug1, bug2, label, s, i))
                    return True
                else:
                    totalIgnored += 1

            d[key] = (n, idx)

    logger.info('Total of positive example duplicated: %d' % totalIgnored)

    return False


def saveDataset(folder, databaseName, label, triplets, pairs):
    tripletsFileName = os.path.join(folder, label + "_" + databaseName + "_triplets_random_1.txt")
    pairsFileName = os.path.join(folder, label + "_" + databaseName + "_pairs_random_1.txt")
    saveTriplets(triplets, tripletsFileName)
    savePairs(pairs, pairsFileName)


def createRecallRateDataset(path, masterSet, pairs, info):
    logger = logging.getLogger()
    allBugs = set()
    duplicateBugs = set()
    for pair in pairs:
        b1 = str(pair[0])
        b2 = str(pair[1])

        allBugs.add(b1)
        allBugs.add(b2)

        if pair[2] == -1:
            continue

        if b1 not in  masterSet:
            duplicateBugs.add(b1)


        if b2 not in  masterSet:
            duplicateBugs.add(b2)



    logger.info("Amount of bug reports: %d" % len(allBugs))
    logger.info("Amount of duplicate reports: %d" % len(duplicateBugs))

    f = open(path, 'w')

    f.write(info)
    f.write('\n')

    for bugId in allBugs:
        f.write('%s ' % (bugId))

    f.write('\n')

    for dupId in duplicateBugs:
        f.write('%s ' % (dupId))

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--database', required=True, help="dataset name")
    parser.add_argument('--collection', help="dataset name")
    parser.add_argument('--folder', required=True)
    parser.add_argument('--validation_size', type=float, default=0.2, help="where will save the training data")
    parser.add_argument('--test_size', type=float, default=0.2, help="where will save the training data")
    parser.add_argument('--enable_duplicate_pairs', action='store_true', help="Duplicate pairs that are positive "
        "are enable to occur")
    parser.add_argument('--seed', default=15516417)

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logHandler = logging.StreamHandler()
    logger.addHandler(logHandler)

    logFile = logging.FileHandler(os.path.join(args.folder, 'dataset.log'))
    logger.addHandler(logFile)

    logger.info(args)

    if args.enable_duplicate_pairs:
        logger.info("########################################")
        logger.info("WARNING: You allow to have duplicate pairs(positive class) in the datasets!!!!!!")
        logger.info("########################################\n")

    logger.info("Creating pairs")
    if args.collection is None:
        args.collection = args.database

    client = pymongo.MongoClient()
    database = client[args.database]

    # Info
    info = "Database: %s; " % args.database
    info += "Not order by date; "
    info += "Deshmukh methodology;"
    info += "We merged the nested master reports;"
    info += "Enable duplicate pairs: %s ;" % args.enable_duplicate_pairs
    info += "Seed: %s ;" % args.seed

    bugPerId, masterSet, masterByBugId, mastersetByMasterId = getMasterSetAndDuplicateByMaster(args.database, args.collection)

    allPairsDict = {}
    if 'pairs' in database.collection_names():
        logger.info("Pairs collection already exist!")
        duplicatePairs = getAllDuplicatePairs(args.database, allPairsDict, args.enable_duplicate_pairs)
    else:
        logger.info("Creating duplicate pairs.")
        duplicatePairs = generateAllDuplicatePairs(args.database, mastersetByMasterId, allPairsDict)

    logger.info("Creating training and test dataset")

    randomSeed = np.random.RandomState(args.seed)

    trainDuplicatePairs, testDuplicatePairs = train_test_split(duplicatePairs, test_size=args.test_size, random_state=randomSeed)
    trainSplitDuplicatePairs, validationDuplicatePairs = train_test_split(trainDuplicatePairs, test_size=args.validation_size, random_state=randomSeed)

    random.seed(args.seed)
    np.random.seed(args.seed)

    total = len(duplicatePairs)

    proportion = 1

    logger.info(
        "Training duplicate pairs: %d (%.2f); Validation duplicate pairs: %d (%.2f); Test duplicate pairs: %d (%.2f)" %
        (len(trainSplitDuplicatePairs), len(trainSplitDuplicatePairs) / total * 100,
         len(validationDuplicatePairs), len(validationDuplicatePairs) / total * 100,
         len(testDuplicatePairs), len(testDuplicatePairs) / total * 100))

    masterList = list(masterSet)

    logger.info('--------------------------------------------')
    logger.info("Generating training triplets and training pairs")
    trainingTriplets = generateTriplets(trainSplitDuplicatePairs, masterList, proportion, allPairsDict)
    trainingPairs = tripletToPairs(trainingTriplets, args.enable_duplicate_pairs)

    training_split = os.path.join(args.folder, 'validation')
    saveDataset(args.folder, args.database, 'training_split', trainingTriplets, trainingPairs)

    logger.info("Training pairs: %d; Training triplets : %d" % (len(trainingPairs), len(trainingTriplets)))

    logger.info('--------------------------------------------')
    logger.info("Generating validation pairs")
    validationTriplets = generateTriplets(validationDuplicatePairs, masterList, proportion, allPairsDict)
    validationPairs = tripletToPairs(validationTriplets, args.enable_duplicate_pairs)

    saveDataset(args.folder, args.database, 'validation_', validationTriplets, validationPairs)
    logger.info("Validation pairs: %d; Validation triplets : %d" % (len(validationPairs), len(validationTriplets)))

    logger.info('--------------------------------------------')
    logger.info("Generating test pairs")
    testTriplets = generateTriplets(testDuplicatePairs, masterList, proportion, allPairsDict)
    testPairs = tripletToPairs(testTriplets, args.enable_duplicate_pairs)

    saveDataset(args.folder, args.database, 'test', testTriplets, testPairs)
    logger.info("Test pairs: %d; Test triplets : %d" % (len(testPairs), len(testTriplets)))

    logger.info('---------------------------------------------------------------')
    logger.info("Checking if there are repeated pairs.")

    if hasDuplicatePairs(trainingPairs, validationPairs, testPairs, args.enable_duplicate_pairs):
        raise Exception('There are duplicate pairs in the dataset!!')

    # Create training dataset that will use to calculate the recall rate
    logger.info('---------------------------------------------------------------')

    trainingSplitRR = os.path.join(args.folder, "training_split_%s.txt" % args.database)
    logger.info("Generating training dataset RR: %s" % trainingSplitRR)
    createRecallRateDataset(trainingSplitRR, masterSet, trainingPairs, info)

    logger.info('---------------------------------------------------------------')

    # Create validation dataset that will use to calculate the recall rate
    validationRR = os.path.join(args.folder, "validation_%s.txt" % args.database)
    logger.info("Generating validation dataset RR: %s" % validationRR)
    createRecallRateDataset(validationRR, masterSet, validationPairs, info)

    logger.info('---------------------------------------------------------------')

    # Create test dataset that will use to calculate the recall rate
    testRR = os.path.join(args.folder, "test_%s.txt" % args.database)
    logger.info("Generating test dataset RR: %s" % testRR)
    createRecallRateDataset(testRR, masterSet, testPairs, info)

    logger.info('---------------------------------------------------------------')

    # Save bugs in json
    databaseFilePath = os.path.join(args.folder, "{}_{}.json".format(args.database, args.collection))
    logger.info("Saving Dataset: %s" % databaseFilePath)

    databaseFile = open(databaseFilePath, 'w')

    for bugId, bug in sorted(bugPerId.items(), key=lambda x: int(x[0])):
        # Saving bug in txt file
        del bug['_id']
        databaseFile.write(ujson.dumps(bug))
        databaseFile.write('\n')

    logger.info("Finished!!!")
