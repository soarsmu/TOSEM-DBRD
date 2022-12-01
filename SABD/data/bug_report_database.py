"""
This class represents a bug report database where we can find all bug reports that are available.
"""
import codecs
import logging
from collections import OrderedDict

import ujson as js


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
        iterator = map(lambda line: js.loads(line) if len(line.strip()) > 0 else None, f)
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
