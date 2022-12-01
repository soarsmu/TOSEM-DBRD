"""
Created by happygirlzt on 6 Aug 2021
Modified on 1 Dec 2022
"""

import codecs
from re import IGNORECASE
import ujson
import os
from tqdm import tqdm
import re
from modules import BugReportDatabase
import argparse
from shutil import copyfile


def check_all_jsons():
    """
    check all the valid jsons, filtering jsons that not involve vscodebots
    """

    involved_count = 0

    for file_name in tqdm(os.listdir(path)):
        full_name = os.path.join(path, file_name)
        with codecs.open(full_name, 'r', encoding='utf-8') as f:
            cur_json = ujson.loads(f.read())

        for cur_comment in cur_json['comments']['nodes']:
            if cur_comment['author'] and cur_comment['author']['login'] == 'vscodebot':
                copyfile(full_name, os.path.join(vscode_bot_path, file_name))
                involved_count += 1
                break

    print('in total {} issues involved vscode bot'.format(involved_count))

def check_dup_detection():
    """
    Among all the issues involving 'vscodebot', filtering those predict duplicates
    the relevant issues are saved in our_data/vscodebot_involved
    """

    removed_count = 0
    with codecs.open(vscode_bot_predictions, 'w', 'utf-8') as jsonFile:

        for file_name in tqdm(os.listdir(vscode_bot_path)):
            bug_id = file_name.split('.')[0]

            full_name = os.path.join(vscode_bot_path, file_name)
            with codecs.open(full_name, 'r', encoding='utf-8') as f:
                cur_json = ujson.loads(f.read())

            find = False
            for cur_comment in cur_json['comments']['nodes']:
                if cur_comment['author'] and cur_comment['author']['login'] == 'vscodebot':
                    cur_text = cur_comment['body']
                    if len(re.findall(r'Experimental duplicate detection', cur_text, IGNORECASE)) > 0:
                        jsonFile.write(ujson.dumps({bug_id: cur_text}))
                        jsonFile.write('\n')

                        find = True
                        break

            if not find:
                os.remove(full_name)
                removed_count += 1

    print('count of not involving dup detection: {}'.format(removed_count))


def detect_vscode_prediction():
    """
    vscode_bot_predictions: all the 11,808 issues and vscodebot's dup prediction in comments
    vscodebot_predictions: all the 11,808 issues and vscodebot's dup detection with numbers
    """

    with codecs.open(vscode_bot_predictions, 'r', 'utf-8') as f:
        pred_lines = f.readlines()

    with codecs.open(vscodebot_predictions, 'w', 'utf-8') as f:
        for pred_line in tqdm(pred_lines):
            cur_prediction = ujson.loads(pred_line)

            bug_id = list(cur_prediction.keys())[0]
            cur_pred = []
            
            for item in re.findall(r'\(#\d+\)', cur_prediction[bug_id], re.IGNORECASE):
                cur_pred.append(re.findall(r'\d+', item, re.IGNORECASE)[0])

            f.write(ujson.dumps({'bug_id': bug_id, 'bot_pred': cur_pred}))
            f.write('\n')


def write_to_file(cont_list, file_name):
    with codecs.open(file_name, 'w', 'utf-8') as f:
        for item in cont_list:
            f.write(str(item))
            f.write('\n')


def compare_with_ground_truth():
    bot_pred_no_groud_truth = []
    bot_pred_hit_positions = []
    bot_pred_missed = []
    dup_bot_no_pred = []

    with codecs.open(vscodebot_predictions, 'r', 'utf-8') as f:
        pred_lines = f.readlines()

    with codecs.open(latest_vscode, 'r', 'utf-8') as f:
        content_lines = f.readlines()

    bug_dups = {}
    for line in content_lines:
        cur_dict = ujson.loads(line)
        if 'dup_id' in cur_dict and len(cur_dict['dup_id']) > 0:
            bug_dups[cur_dict['bug_id']] = cur_dict['dup_id']

    overlapped = 0
    total_preds = set()
    
    for line in tqdm(pred_lines):
        cur_dict = ujson.loads(line)
        cur_bug_id = cur_dict['bug_id']
        total_preds.add(cur_bug_id)
        
        if cur_bug_id in bug_dups:
            overlapped += 1
            cur_groud_truth = bug_dups[cur_bug_id]
            # print(cur_groud_truth)
            found = False

            for i in range(len(cur_dict['bot_pred'])):

                if cur_dict['bot_pred'][i] == cur_groud_truth:
                    bot_pred_hit_positions.append((cur_bug_id, i))
                    found = True
                    break

            if not found:
                # print(cur_bug_id)
                bot_pred_missed.append(cur_bug_id)
        else:
            bot_pred_no_groud_truth.append(cur_bug_id)
            
    dup_bot_no_pred = bug_dups.keys() - total_preds
    
    print('total overlapped {}'.format(overlapped))
    print('total dups and bot did not predict {}'.format(len(dup_bot_no_pred)))
    print('total has no ground truth {}'.format(len(bot_pred_no_groud_truth)))
    print('total hits {}'.format(len(bot_pred_hit_positions)))
    print('total missed {}'.format(len(bot_pred_missed)))

    write_to_file(dup_bot_no_pred, 'dup_bot_not_pred.txt')
    write_to_file(bot_pred_no_groud_truth, 'bot_pred_no_ground_truth.txt')
    write_to_file(bot_pred_hit_positions, 'bot_pred_hit_positions.txt')
    write_to_file(bot_pred_missed, 'bot_pred_missed.txt')


def compare_with_ground_truth_within_test():
    bugReportDatabase = BugReportDatabase.fromJson(latest_vscode)
    masterSetById = bugReportDatabase.getMasterSetById()
    masterIdByBugId = bugReportDatabase.getMasterIdByBugId()

    with codecs.open(latest_vscode, 'r', 'utf-8') as f:
        content_lines = f.readlines()

    with codecs.open('../SABD/dataset/vscode/test_vscode.txt', 'r') as f:
        lines = f.readlines()
        dup_test_ids = lines[2].split(' ')

    bug_dups = list()
    bug_dups_set = set()
    
    # test set only contains 3k BRs, and only 1457 BRs are duplicate
    # among them, vscodebot predicted 297 BRs
    
    for line in content_lines:
        cur_dict = ujson.loads(line)
        if cur_dict['bug_id'] in dup_test_ids:
            bug_dups.append({cur_dict['bug_id'] : cur_dict['dup_id']})
    
    for bug_dup_pair in bug_dups:
        bug_dups_set.add(list(bug_dup_pair.keys())[0])

    overlapped = 0

    bot_predictions_all = {}

    hits = [0] * 5

    with codecs.open(vscodebot_predictions, 'r', 'utf-8') as f:
        pred_lines = f.readlines()

    for line in tqdm(pred_lines):
        cur_dict = ujson.loads(line)
        cur_bug_id = cur_dict['bug_id']
        bot_predictions_all[cur_bug_id] = cur_dict['bot_pred']
        
    print(len(bot_predictions_all))

    need_to_check = [100] * len(bug_dups)

    missed = 0
    for i in tqdm(range(len(bug_dups))):
        cur_bug_id = list(bug_dups[i].keys())[0]

        if cur_bug_id in bot_predictions_all:
            overlapped += 1

            found = False

            # for j in range(len(bot_predictions_all[cur_bug_id])):
            # print(cur_bug_id)
            mastersetId = masterIdByBugId[cur_bug_id]
            masterSet = masterSetById[mastersetId]
            seenMasters = set()

            for bugId in bot_predictions_all[cur_bug_id]:
                try:
                    mastersetId = masterIdByBugId[bugId]

                    if mastersetId in seenMasters:
                        continue

                    seenMasters.add(mastersetId)

                    if bugId in masterSet:
                        pos = len(seenMasters)
                        need_to_check[i] = pos - 1
                        hits[pos - 1] += 1
                        found = True
                        break
                except KeyError:
                    seenMasters.add(-len(seenMasters)-1)

            # if bot_predictions_all[cur_bug_id][j] == cur_groud_truth:
                # need_to_check[i] = j
                # hits[j] += 1
                # found = True
                # break

            if not found:
                missed += 1
                need_to_check[i] = float('inf')
                
    os.makedirs('./result', exist_ok=True)
    
    write_to_file(need_to_check, './result/need_to_check_1.txt')
    print(hits)

    for i in range(1, len(hits)):
        hits[i] += hits[i - 1]

    recall_rate = [0] * 5
    for i in range(len(hits)):
        recall_rate[i] = round(hits[i] / 297, 2)

    print('recall rate {}'.format(recall_rate))
    print('total hits {}'.format(hits[-1]))
    print('missed {}'.format(missed))
    print('total overlapped {}'.format(overlapped))


def analyze_hits():
    hit_file = './result/bot_pred_hit_positions.txt'
    with codecs.open(hit_file, 'r') as f:
        lines = f.readlines()

    position_count = [0, 0, 0, 0, 0]

    for line in lines:
        # print(line[1:-2])
        # print(line[1:-2].split(',')[1].strip())
        position_count[int(line[1:-2].split(',')[1].strip())] += 1

    print(position_count)


def analyze_one_time_predictions():
    # we need to copy and paste the predicted positions under result folder
    with open('./result/{}.txt'.format(approach)) as f:
        pred_positions = f.readline().split(',')
    
    with open('./result/need_to_check_1.txt') as f:
        lines = f.readlines()

    hits = [0, 0, 0, 0, 0]

    missed = 0
    for i in range(len(lines)):
        bot_pred = lines[i].strip()
        if bot_pred == '100':
            continue

        if pred_positions[i].strip() == 'inf':
            missed += 1
            continue

        cur_pred = int(pred_positions[i].strip())

        if cur_pred <= 5:
            hits[cur_pred - 1] += 1
        else:
            missed += 1

    print(approach)
    print("hits: ", hits)

    for i in range(1, len(hits)):
        hits[i] += hits[i - 1]

    recall_rate = [0] * 5
    for i in range(len(hits)):
        recall_rate[i] = round(hits[i] / 297, 2)

    print('recall rate {}'.format(recall_rate))
    print('total hits {}'.format(hits[-1]))
    print('missed {}'.format(missed))


def get_averaged_predictions(approach):
    """
    if run several times, get the averaged predicted positions
    """
    
    five_pred_file = './result/{}.txt'.format(approach)
    with open(five_pred_file) as f:
        lines = f.readlines()

    positions = [0] * 5

    for i in range(5):
        positions[i] = lines[i].split(',')

    average_positions = list()

    for i in tqdm(range(len(positions[0]))):
        total_positions = 0
        has_inf = False

        for j in range(5):
            cur_pred = positions[j][i].strip()

            if cur_pred == 'inf':
                has_inf = True
                break
            else:
                total_positions += int(cur_pred)

        if not has_inf:
            average_positions.append(str(total_positions // 5))
        else:
            average_positions.append(str('inf'))

    with open('./result/avg_{}.txt'.format(approach), 'w') as f:
        for position in average_positions:
            f.write(position)
            f.write('\n')


def analyze_averaged_predictions():
    """
    analyze the averaged performance
    """
    
    os.makedirs('./result/', exist_ok=True)
    
    with open('./result/avg_{}.txt'.format(approach)) as f:
        avg_lines = f.readlines()
    
    with open('./result/need_to_check_1.txt') as f:
        lines = f.readlines()

    hits = [0, 0, 0, 0, 0]

    missed = 0
    for i in range(len(lines)):
        bot_pred = lines[i].strip()
        if bot_pred == '100':
            continue

        if avg_lines[i].strip() == 'inf':
            missed += 1
            continue

        cur_pred = int(avg_lines[i].strip())

        if cur_pred <= 5:
            hits[cur_pred - 1] += 1
        else:
            missed += 1

    print(approach)
    print(hits)

    for i in range(1, len(hits)):
        hits[i] += hits[i - 1]

    recall_rate = [0] * 5
    for i in range(len(hits)):
        recall_rate[i] = hits[i] / 297

    print('recall rate {}'.format(recall_rate))

    print('hits {}'.format(hits))

    print('total hits {}'.format(sum(hits)))
    print('missed {}'.format(missed))


def check_rep():
    with open('./result/avg_rep.txt') as f:
        rep_lines = f.readlines()
    
    with open('./result/need_to_check_1.txt') as f:
        lines = f.readlines()

    hits = [0, 0, 0, 0, 0]
    missed = 0
    for i in range(len(lines)):
        bot_pred = lines[i].strip()
        if bot_pred == '100':
            continue

        try:
            cur_pred = int(rep_lines[i].strip())
            if cur_pred <= 4:
                hits[cur_pred] += 1
            else:
                missed += 1
        except ValueError:
            missed += 1
            pass
    print(hits)
    for i in range(1, len(hits)):
        hits[i] += hits[i - 1]

    recall_rate = [0] * 5
    for i in range(len(hits)):
        recall_rate[i] = round(hits[i] / 297, 2)

    print('recall rate {}'.format(recall_rate))
    print('total hits {}'.format(hits[-1]))
    print('missed {}'.format(missed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--approach', help='project name')

    args = parser.parse_args()
    approach = args.approach
    
    path = '../../our_data/vscode'
    vscode_bot_path = './../our_data/vscodebot_involved'
    vscode_bot_predictions = '../../our_data/bot_predictions.json'
    vscodebot_predictions = '../../our_data/vscodebot_predictions.json'
    latest_vscode = '../SABD/dataset/vscode/vscode.json'


    # check_all_jsons()
    # check_dup_detection()
    # detect_vscode_prediction()
    # compare_with_ground_truth()
    # analyze_hits()
    # analyze_vscode_predictions()
    # compare_with_ground_truth_within_test()
    # get_averaged_predictions(approach)
    # analyze_averaged_predictions()
    check_rep()
    # analyze_one_time_predictions()