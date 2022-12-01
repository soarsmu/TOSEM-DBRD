import argparse
import logging
import pickle
import random

from data.bug_dataset import BugDataset
from data.bug_report_database import BugReportDatabase

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--bug_data', required=True, help="")
    parser.add_argument('--dataset', required=True, help="")
    parser.add_argument('--output', required=True, help="")
    parser.add_argument('--size', default=40000, type=int, help="")
    parser.add_argument('--only_master', action="store_true",
                        help="Only compare the new bugs with the master sets.")
    parser.add_argument('--random', action="store_true",
                        help="Get randomly the n bug of the list.")

    parser.add_argument('--random_date', action="store_true",
                        help="Randomly select reports that were reported before a specific bug report.")

    parser.add_argument('--sample', default=0, type=int, help="Sample duplciate.")

    logging.basicConfig(level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    args = parser.parse_args()
    logger.info(args)

    bugDataset = BugDataset(args.dataset)
    bugReportDatabase = BugReportDatabase.fromJson(args.bug_data)

    bugIds = set(bugDataset.bugIds)
    allDuplicateBugs = bugDataset.duplicateIds

    if args.sample > 0:
        allDuplicateBugs = random.sample(bugDataset.duplicateIds, args.sample)

    if not args.random:
        # If it isn't random, so we consider that the bugs are sorted by the creation date in BugDataset and Database
        positionIdByBugId = {}

        for idx, bug in enumerate(bugReportDatabase.bugList):
            positionIdByBugId[bug['bug_id']] = idx

    # Insert all master ids to the bug id list
    masterIdByBugId = bugReportDatabase.getMasterIdByBugId()
    masterSetByBugId = bugReportDatabase.getMasterSetById(bugIds)

    masterNotFound = 0
    setOfBugIds = set()
    listByBugId = {}

    lastBugId = bugDataset.bugIds[-1]

    for duplicateBugId in allDuplicateBugs:
        dupMasterId = masterIdByBugId[duplicateBugId]
        bugToTest = []
        duplicateOfBug = []

        if args.random:
            # Retrieve all the bug reports that have an id smaller than duplicateBugId
            candidates = random.sample(bugIds, args.size)
        else:
            duplicateIdx = positionIdByBugId[duplicateBugId]
            idxs = range(duplicateIdx - 1, -1, -1)

            if args.random_date:
                idxs = list(idxs)
                random.shuffle(idxs)

            candidates = map(lambda idx: bugReportDatabase.getBugByIndex(idx)['bug_id'], idxs)

        for bugId in candidates:
            if len(bugToTest) == args.size:
                break

            masterId = masterIdByBugId[bugId]

            if args.only_master and masterId != bugId:
                continue

            bugToTest.append(bugId)

            if dupMasterId == masterId:
                duplicateOfBug.append(bugId)

        if len(bugToTest) < args.size:
            logger.warning("We only found %d bugs for the bug %s" % (len(bugToTest), duplicateBugId))

        # Guarantee that there is at only one duplicate bug in the list
        if len(duplicateOfBug) == 0:
            bugToTest.pop()
            if args.random:
                masterSet = masterSetByBugId[dupMasterId]

                otherBugId = None
                for k in masterSet:
                    if k in bugIds:
                        otherBugId = k
                        break
                if otherBugId is None:
                    raise Exception('We did not find any of duplicate bugs of %s' % duplicateBugId)

                bugToTest.append(otherBugId)
            else:
                bugToTest.append(dupMasterId)
            masterNotFound += 1

        setOfBugIds.update(bugToTest)
        setOfBugIds.add(duplicateBugId)

        listByBugId[duplicateBugId] = bugToTest

    # Check if is correct
    for bugId, bugs in listByBugId.items():
        masterId = masterIdByBugId[bugId]
        nDup = 0

        for other in bugs:
            if masterIdByBugId[other] == masterId:
                nDup += 1

        if nDup == 0:
            raise Exception("{} list doesn't have any duplicate".format(bugId))

    setOfBugIds = list(setOfBugIds)
    pickle.dump([setOfBugIds, listByBugId], open(args.output, 'wb'))

    logger.info("The master was not found in %d bug of %d" % (masterNotFound, len(allDuplicateBugs)))
