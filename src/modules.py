import codecs
from collections import OrderedDict
import ujson
import logging
from datetime import datetime


class BugReportDatabase(object):
    '''

    Load bug report data (categorical information, summary and description) from json file.
    '''

    def __init__(self, iterator):
        self.bugById = OrderedDict()
        self.bugList = []
        self.logger = logging.getLogger()

        nEmptyDescription = 0

        for bug in iterator:
            if bug is None:
                continue

            bugId = bug["bug_id"]

            self.bugById[bugId] = bug
            self.bugList.append(bug)

            description = bug["description"]

            if isinstance(description, list) or len(description.strip()) == 0:
                nEmptyDescription += 1

        self.logger.info("Number of bugs with empty description: %d" % nEmptyDescription)

    @staticmethod
    def fromJson(fileToLoad):
        f = codecs.open(fileToLoad, 'r', encoding='utf-8')
        iterator = map(lambda line: ujson.loads(line) if len(line.strip()) > 0 else None, f)
        return BugReportDatabase(iterator)

    def getBug(self, bugId):
        return self.bugById[bugId]

    def getBugByIndex(self, idx):
        return self.bugList[idx]

    def __len__(self):
        return len(self.bugList)
    
    def __contains__(self, bug):
        bugId = bug['bug_id'] if isinstance(bug, dict) else bug

        return bugId in self.bugById

    def getMasterIdByBugId(self, bugs=None):
        # return the dup_id by bug_id
        masterIdByBugId = {}
        bugs = self.bugList if bugs is None else bugs

        for bug in bugs:
            if not isinstance(bug, dict):
                bug = self.bugById[bug]

            bugId = bug['bug_id']
            dupId = bug['dup_id']

            if len(dupId) != 0:
                masterIdByBugId[bugId] = dupId
            else:
                masterIdByBugId[bugId] = bugId

        return masterIdByBugId

    def getMasterSetById(self, bugs=None):
        masterSetById = {}
        bugs = self.bugList if bugs is None else bugs

        for bug in bugs:
            if not isinstance(bug, dict):
                bug = self.bugById[bug]

            dupId = bug['dup_id']

            if len(dupId) != 0:
                masterSet = masterSetById.get(dupId, set())

                if len(masterSet) == 0:
                    masterSetById[dupId] = masterSet

                masterSet.add(bug['bug_id'])

        # Insert id of the master bugs in your master sets
        for masterId, masterSet in masterSetById.items():
            if masterId in self:
                masterSet.add(masterId)

        return masterSetById

class BugDataset(object):
    def __init__(self, file):
        f = open(file, 'r')
        self.info = f.readline().strip()
        self.bugIds = [id for id in f.readline().strip().split()]
        self.duplicateIds = [id for id in f.readline().strip().split()]

class SunRanking(object):

    def __init__(self, bugReportDatabase, dataset, window):
        self.bugReportDatabase = bugReportDatabase
        self.masterIdByBugId = self.bugReportDatabase.getMasterIdByBugId()
        self.duplicateBugs = dataset.duplicateIds
        self.candidates = []
        self.window = int(window) if window is not None else 0
        self.latestDateByMasterSetId = {}
        self.logger = logging.getLogger()

        # Get oldest and newest duplicate bug report in dataset
        oldestDuplicateBug = (
            self.duplicateBugs[0], 
            readDateFromBug(self.bugReportDatabase.getBug(self.duplicateBugs[0]))
        )

        for dupId in self.duplicateBugs:
            dup = self.bugReportDatabase.getBug(dupId)
            creationDate = readDateFromBug(dup)
            
            # 这其实找的是最新的bug
            if oldestDuplicateBug[1] < creationDate:
                oldestDuplicateBug = (dupId, creationDate)
                
        # 所以只要bug比最新的bug要老，就可能是candidate
        # Keep only master that are able to be candidate
        for bug in self.bugReportDatabase.bugList:
            bugCreationDate = readDateFromBug(bug)
            bugId = bug['bug_id']

            # Remove bugs that their creation time is bigger than oldest duplicate bug
            if bugCreationDate > oldestDuplicateBug[1] or (
                    bugCreationDate == oldestDuplicateBug[1] and bug['bug_id'] > oldestDuplicateBug[0]):
                continue

            self.candidates.append((bugId, bugCreationDate.timestamp()))

        # Keep the timestamp of all reports in each master set
        for masterId, masterSet in self.bugReportDatabase.getMasterSetById(
                map(lambda c: c[0], self.candidates)).items():
            ts_list = []

            for bugId in masterSet:
                bugCreationDate = readDateFromBug(self.bugReportDatabase.getBug(bugId))

                ts_list.append((int(bugId), bugCreationDate.timestamp()))
                
            # latestDateByMasterSetId里面存的是一个masterId和所有的candidates
            self.latestDateByMasterSetId[masterId] = ts_list

        # Set all bugs that are going to be used by our models.
        self.allBugs = [bugId for bugId, bugCreationDate in self.candidates]
        self.allBugs.extend(self.duplicateBugs)

    def getDuplicateBugs(self):
        return self.duplicateBugs

    def getAllBugs(self):
        return self.allBugs

    def getCandidateList(self, anchorId):
        # 这段是检查，如果self.candidates里的符合条件就作为
        # 这个anchorID的真candidates
        candidates = []
        anchor = self.bugReportDatabase.getBug(anchorId)
        anchorCreationDate = readDateFromBug(anchor)
        anchorMasterId = self.masterIdByBugId[anchorId]
        nDupBugs = 0
        anchorTimestamp = anchorCreationDate.timestamp()
        anchorDayTimestamp = int(anchorTimestamp / (24 * 60 * 60))

        nSkipped = 0
        window_record = [] if self.logger.isEnabledFor(logging.DEBUG) else None
        anchorIdInt = int(anchorId)

        for bugId, bugCreationDate in self.candidates:
            bugIdInt = int(bugId)

            # Ignore reports that were created after the anchor report
            if bugCreationDate > anchorTimestamp or (
                    bugCreationDate == anchorTimestamp and bugIdInt > anchorIdInt):
                continue
            # Check if the same report
            if bugId == anchorId:
                continue
            
            # 主要还是看creationDate，不是对比id的
            if bugIdInt > anchorIdInt:
                self.logger.warning(
                    "Candidate - consider a report which its id {} is bigger than duplicate {}".format(bugId, anchorId)
                )

            masterId = self.masterIdByBugId[bugId]

            # Group all the duplicate and master in one unique set. Creation date of newest report is used to filter the bugs
            tsMasterSet = self.latestDateByMasterSetId.get(masterId)
            
            # 得到创建时间最近的bug
            if tsMasterSet:
                max = -1
                newest_report = None

                for candNewestId, ts in self.latestDateByMasterSetId[masterId]:
                    # Ignore reports that were created after the anchor or the ones that have the same ts and bigger id
                    if ts > anchorTimestamp or (ts == anchorTimestamp and candNewestId >= anchorIdInt):
                        continue

                    if candNewestId >= anchorIdInt:
                        self.logger.warning(
                            "Window filtering - consider a report which its id {} is bigger than duplicate {}".format(candNewestId, anchorIdInt)
                        )

                    # Get newest ones
                    if max < ts:
                        max = ts
                        newest_report = candNewestId

                # Transform to day timestamp
                bug_timestamp = int(max / (24 * 60 * 60))
            else:
                # Transform to day timestamp
                bug_timestamp = int(bugCreationDate / (24 * 60 * 60))
                newest_report = bugId

            # Is it in the window?
            if 0 < self.window < (anchorDayTimestamp - bug_timestamp):
                nSkipped += 1
                continue

            # Count number of duplicate bug reports
            if anchorMasterId == masterId:
                nDupBugs += 1

            # It is a candidate
            candidates.append(bugId)
            if window_record is not None:
                window_record.append((bugId, newest_report, bug_timestamp))

        self.logger.debug(
            "Query {} ({}) - window {} - number of reports skipped: {}".format(anchorId, anchorDayTimestamp, self.window, nSkipped)
        )

        if window_record is not None:
            self.logger.debug("{}".format(window_record))

        if nDupBugs == 0:
            print(anchorId)
            return []

        return candidates

def readDateFromBug(bug):
    return datetime.strptime(bug['creation_ts'], '%Y-%m-%d %H:%M:%S %z')