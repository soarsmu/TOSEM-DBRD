"""
Created by AnonymousAuthor
on 11 Aug 2021

extracting the successful predictions from REP on vscode project
"""

import re
from tqdm import tqdm

rep_result_1 = '../REP/recommend_ranknet_29_Dec_vscode_I-1'

rep_results = [
    '../REP/recommend_ranknet_22_Aug_2022_vscode-1_I-1',
    '../REP/recommend_ranknet_22_Aug_2022_vscode-2_I-1',
    '../REP/recommend_ranknet_22_Aug_2022_vscode-3_I-1',
    '../REP/recommend_ranknet_22_Aug_2022_vscode-4_I-1',
    '../REP/recommend_ranknet_22_Aug_2022_vscode-5_I-1',
]

def get_averaged_prediction():
    """
    get averaged predictions by several times
    """
    
    positions = [0] * 5
    for index, res_file in zip(range(len(rep_results)), rep_results):        
        with open(res_file) as f:
            lines = f.readlines()
        
        for l in tqdm(range(len(lines))):
            line = lines[l]
            
            iteration = re.findall(r'Iteration \d', line)
            if len(iteration) > 0:
                pred = [float('inf')] * 1457

                for i in range(l + 1, len(lines), 22):
                    if lines[i].strip() == '':
                        break

                    for j in range(i + 1, i + 6):
                        if lines[j].strip()[-1] == '+':
                            pred[(i - l) // 22] = j - i - 1
                            break
                        
                positions[index] = pred

    # calculate the average positions
    average_positions = list()
    for i in tqdm(range(len(positions[0]))):
        total_positions = 0
        has_inf = False

        for j in range(5):
            cur_pred = positions[j][i]

            if cur_pred == float('inf'):
                has_inf = True
                break
            else:
                total_positions += int(cur_pred)

        if not has_inf:
            average_positions.append(str(total_positions // 5))
        else:
            average_positions.append(str('inf'))

    with open('./result/avg_rep.txt', 'w') as f:
        for position in average_positions:
            f.write(position)
            f.write('\n')



def get_onetime_prediction():
    with open(rep_result_1) as f:
        lines = f.readlines()

    positions = list()
    for l in tqdm(range(len(lines))):
        line = lines[l]

        iteration = re.findall(r'Iteration \d', line)
        if len(iteration) > 0:

            pred = [float('inf')] * 1457

            for i in range(l + 1, len(lines), 22):
                if lines[i].strip() == '':
                    break

                for j in range(i + 1, i + 6):
                    if lines[j].strip()[-1] == '+':
                        pred[(i - l) // 22] = j - i - 1
                        break
            
            positions = pred


    with open('./result/rep.txt', 'w') as f:
        for position in positions:
            if position == float('inf'):
                f.write(str('inf'))
            else:
                f.write(str(position))
            f.write('\n')


if __name__ == '__main__':
    # get_onetime_prediction()
    get_averaged_prediction()