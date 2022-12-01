import argparse
import json

from data.bug_report_database import BugReportDatabase

parser = argparse.ArgumentParser()
parser.add_argument('--bug_dataset', required=True, help="")
parser.add_argument('--file', help="")

args = parser.parse_args()

bugReportDataset = BugReportDatabase.fromJson(args.bug_dataset)

f = open(args.file, "r")
l = json.load(f)


i = 0
for bugId1, bugId2, target, prediction, summary1, summary2 in l:
    bug1 = bugReportDataset.getBug(bugId1)
    bug2 = bugReportDataset.getBug(bugId2)

    print("-------------------------------------")
    print('Bug Id: %s\t%s' % (bugId1, bugId2))
    print('Target: %d\tPrediction: %d' % (target, prediction))
    print('Product: %s\t%s' % (bug1['product'], bug2['product']))
    print('Bug Severity: %s\t%s' % (bug1['bug_severity'], bug2['bug_severity']))
    print('Component: %s\t%s' % (bug1['component'], bug2['component']))
    print('Priority: %s\t%s' % (bug1['priority'], bug2['priority']))
    print('Version: %s\t%s' % (bug1['version'], bug2['version']))
    print("Component: %s\t%s" % (bug1['component'], bug2['component']))
    print("")
    print(bug1['short_desc'])
    print(bug2['short_desc'])
    print("")
    print(summary1)
    print(summary2)
    print("\n")
    if i != 0 and i == 20:
        input("Read more 20 wrong predictions?")
        i = 0

    i+=1



