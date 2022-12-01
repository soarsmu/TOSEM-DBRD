import sys, os
sys.path.append('../SABD/')
from data.bug_report_database import BugReportDatabase
from statistics import mean
import requests
import re
from tqdm import tqdm


data_folder = '../SABD/dataset/'
# projects = ['eclipse', 'eclipse-old',
#             'mozilla', 'mozilla-old',
#             'hadoop', 'spark',
#             'kibana', 'vscode']

projects = ['mozilla']

def fill_table():
    for project in projects:
        with open(data_folder + '{}/training_{}.txt'.format(project, project)) as f:
            lines = f.readlines()
        count_BRs_train = len(lines[1].strip().split())
        count_dups_train = len(lines[2].strip().split())
        dup_rate_train = count_dups_train / count_BRs_train
        print('{} training: # of BRs {}; # of dups {}; \% of dups: {}'.format(project, count_BRs_train, count_dups_train, dup_rate_train))
        
        with open(data_folder + '{}/training_{}_pairs_random_1.txt'.format(project, project))as f1:
            lines = f1.readlines()
        print('{}: # of dup pairs {}'.format(project, len(lines) / 2))    

        with open(data_folder + '{}/test_{}.txt'.format(project, project)) as f2:
            lines = f2.readlines()
        count_BRs_test = len(lines[1].strip().split())
        count_dups_test = len(lines[2].strip().split())
        dup_rate_test = count_dups_test / count_BRs_test
        print('{} test: # of BRs {}; # of dups {}; \% of dups: {}'.format(project, count_BRs_test, count_dups_test, dup_rate_test))
        total_BRs = count_BRs_train + count_BRs_test
        total_dups = count_dups_train + count_dups_test
        dup_percentage = total_dups / total_BRs
        print('total dups: {}; percentage: {}'.format(total_dups, dup_percentage))
        print('total BRs: {}'.format(total_BRs))
        print('='*20)
        

    project = 'mozilla-old'
    with open(data_folder + '{}/archive/training_{}.txt'.format(project, project)) as f:
        lines = f.readlines()
        count_BRs = len(lines[1].strip().split())
        count_dups = len(lines[2].strip().split())
        print('{}: # of BRs {}; # of dups {}'.format(project, count_BRs, count_dups))
        
        with open(data_folder + '{}/archive/training_{}_pairs_random_1.txt'.format(project, project))as f1:
            lines = f1.readlines()
        print('{}: # of dup pairs {}'.format(project, len(lines) / 2))

        with open(data_folder + '{}/test_{}.txt'.format(project, project)) as f2:
            lines = f2.readlines()
        count_BRs = len(lines[1].strip().split())
        count_dups = len(lines[2].strip().split())
        print('{} test: # of BRs {}; # of dups {}'.format(project, count_BRs, count_dups))
        
def count_master():
    for project in projects:
        print('Project: {}'.format(project))
        
        bugReportDatabase = BugReportDatabase.fromJson(data_folder + '{}/{}_soft_clean.json'.format(project, project))
        
        master_sets = bugReportDatabase.getMasterSetById()
        print(len(bugReportDatabase.bugList))
        # print(master_sets)
        print('unique masters: {}'.format(len(master_sets)))
        
        count = []
        for k, v in master_sets.items():
            if len(v) == 151:
                print(k, v)
            
        print('min: {}'.format(min(count)))
        print('max: {}'.format(max(count)))
        print('mean: {}'.format(mean(count)))
        print('='*20)
        break
    
    project = 'mozilla-old'
    print('Project: {}'.format(project))
    bugReportDatabase = BugReportDatabase.fromJson(data_folder + '{}/archive/{}_soft_clean.json'.format(project, project))
    
    master_sets = bugReportDatabase.getMasterSetById()
    print(len(bugReportDatabase.bugList))
    # print(master_sets)
    print('unique masters: {}'.format(len(master_sets)))
    
    count = []
    for k, v in master_sets.items():
        count.append(len(v))
        if len(v) > max(count):
            print(master_sets)
            
    print('min: {}'.format(min(count)))
    print('max: {}'.format(max(count)))
    print('mean: {}'.format(mean(count)))
    print('='*20)
        
def check_whether_duplicate():
    master = '1427728'
    dups = {'1474161', '1504419', '1472912', '1460135', '1575416', '1473316', '1628909', '1509659', '1470134', '1480629', '1596617', '1492334', '1478200', '1452790', '1479472', '1503069', '1637522', '1582277', '1442991', '1647414', '1570480', '1474136', '1483266', '1488197', '1427728', '1576012', '1462747', '1473350', '1509660', '1539744', '1452871', '1474144', '1594543', '1508906', '1459799', '1474782', '1472492', '1500406', '1577423', '1446103', '1460138', '1476618', '1499219', '1513899', '1550234', '1626395', '1485410', '1502300', '1584885', '1450843', '1632607', '1499921', '1472588', '1550879', '1519641', '1652363', '1628908', '1462750', '1484441', '1458485', '1474220', '1476834', '1468683', '1460113', '1536771', '1460150', '1507034', '1508444', '1455812', '1458509', '1460153', '1628881', '1456493', '1465601', '1457910', '1450412', '1458115', '1575212', '1523752', '1503474', '1678264', '1470834', '1472070', '1663943', '1575327', '1433606', '1457754', '1470102', '1628511', '1441369', '1472245', '1469434', '1459806', '1438061', '1485165', '1458506', '1452872', '1476120', '1517219', '1536770', '1520009', '1460108', '1484204', '1475923', '1479402', '1467937', '1462245', '1443408', '1464590', '1458451', '1470634', '1662918', '1681069', '1443801', '1458141', '1464731', '1464339', '1541971', '1477657', '1437410', '1484813', '1482819', '1518336', '1652299', '1450665', '1479214', '1471142', '1451144', '1484816', '1472774', '1428590', '1644418', '1452477', '1460479', '1460107', '1596447', '1628899', '1663127', '1443071', '1560944', '1437304', '1682053', '1436753', '1459805', '1517392', '1476471', '1631889', '1662904', '1484729', '1473886', '1441261'}
    url = 'https://bugzilla.mozilla.org/show_bug.cgi?ctype=xml&id='
    oldest_one = ''
    for dup in tqdm(dups):
        cur_url = url + dup
        f = requests.get(cur_url)
        contents = f.text
        dupid = re.findall('<dup_id>(.*?)</dup_id>', contents)[0]
        if not dupid in dups and dupid != oldest_one:
            print('cur: ', dup)
            print('dupid: ', dupid)
            oldest_one = dupid
            
if __name__ == '__main__':
    # count_master()
    fill_table()
    # check_whether_duplicate()