# the circle means the 
# correctly predicted BRs (hits within 10 predictions)

import re
import os, sys
from tqdm import tqdm
# import venn
from matplotlib_venn import venn2, venn2_unweighted
from matplotlib import pyplot as plt

methods = ['sabd', 'pairs', 'dccnn', 'hindbr']
all_methods = ['rep', 'sabd', 'pairs', 'dccnn', 'hindbr']

datasets = ['eclipse', 'mozilla', 'hadoop', 'spark', 'kibana', 'vscode']
# datasets = ['mozilla']

def extract_result_from_rep(rep_result, dataset):
    # print(rep_result)
    with open(rep_result) as f:
        lines = f.readlines()

    # with open(rep_result) as f:
    #     file = f.read()
    # pred = [0] * len(re.findall('Retrieving for duplicate report', file))
    pred = []

    for l in tqdm(range(len(lines))):
        line = lines[l]
        iteration = re.findall(r'Iteration \d', line)
        if len(iteration) > 0:
            count = 0
            for i in range(l + 2, len(lines) + 1, 22):
                if lines[i].strip() == '':
                    continue
                                    
                for j in range(i, i + 20):
                    # print(j)
                    if '+' in lines[j].strip():
                        count += 1
                        pred.append('{}_{}'.format((i - l) // 22, dataset))
                        break
                    
    return len(pred), pred

def extract_result(log_file, dataset):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    position_line = lines[-3]
    num_hits = 0
    hit_list = []
    try:
        for i, item in zip(range(len(re.findall('\[(.*)\]',position_line)[0].split(','))), \
            re.findall('\[(.*)\]',position_line)[0].split(',')):
            if item.strip() == 'inf' or int(item.strip()) > 10:
                continue
            hit_list.append('{}_{}'.format(i, dataset))
            num_hits += 1
        return num_hits, hit_list
    except IndexError:
        position_line = lines[-1]
        for i, item in zip(range(len(re.findall('\[(.*)\]',position_line)[0].split(','))), \
            re.findall('\[(.*)\]',position_line)[0].split(',')):
            if item.strip() == 'inf' or int(item.strip()) > 10:
                continue
            hit_list.append('{}_{}'.format(i, dataset))
            num_hits += 1
        return num_hits, hit_list


def calculate_all_method(method):
    total_hits, total_hit_list = 0, [] 
    log_files = [filename for filename in os.listdir('./drawing/') if filename.startswith("{}".format(method))]

    for dataset in datasets:
        for log_file in log_files:
            if log_file.startswith('{}_{}'.format(method, dataset)):
                print('{}'.format(dataset))
                num_hits, hit_list = extract_result('./drawing/{}'.format(log_file), dataset)
                total_hits += num_hits
                total_hit_list.extend(hit_list)
    print('{}: {}'.format(method, total_hits))
    with open('./drawing/{}_all_hits.txt'.format(method), 'w') as f:
        for item in total_hit_list:
            f.write('{}\n'.format(item))


def get_hits(method):
    with open('./drawing/{}_all_hits.txt'.format(method), 'r') as f:
        lines = f.readlines()
    hit_list = [item.strip() for item in lines]
    return hit_list

def get_venn_diagram_numbers():
    hit_list_rep = get_hits('rep')
    hit_list_sabd = get_hits('sabd')
    # hit_list_pairs = get_hits('pairs')
    # hit_list_dccnn = get_hits('dccnn')
    # hit_list_hindbr = get_hits('hindbr')
    labels = venn.get_labels([set(hit_list_rep), set(hit_list_sabd)], fill=['number'])
    fig, ax = venn.venn5(labels, names=['REP', 'SABD'])
    # labels = venn.get_labels([set(hit_list_rep), set(hit_list_pairs), set(hit_list_sabd), 
    # set(hit_list_dccnn), set(hit_list_hindbr)], fill=['number', 'logic'])
    # fig, ax = venn.venn5(labels, names=['0:REP', '1:Siamese Pair', '2:SABD', '3:DCCNN', '4:HINDBR'])
    # fig.show()
    fig.savefig('rep-sabd.pdf')
    
    hit_list_rep = get_hits('rep')
    hit_list_pairs = get_hits('pairs')
    # hit_list_dccnn = get_hits('dccnn')
    # hit_list_hindbr = get_hits('hindbr')
    labels = venn.get_labels([set(hit_list_rep), set(hit_list_pairs)], fill=['number'])
    fig, ax = venn.venn5(labels, names=['REP', 'Siamese Pair'])
    # labels = venn.get_labels([set(hit_list_rep), set(hit_list_pairs), set(hit_list_sabd), 
    # set(hit_list_dccnn), set(hit_list_hindbr)], fill=['number', 'logic'])
    # fig, ax = venn.venn5(labels, names=['0:REP', '1:Siamese Pair', '2:SABD', '3:DCCNN', '4:HINDBR'])
    # fig.show()
    fig.savefig('rep-pair.pdf')
    
    hit_list_rep = get_hits('rep')
    hit_list_dccnn = get_hits('dccnn')
    # hit_list_hindbr = get_hits('hindbr')
    labels = venn.get_labels([set(hit_list_rep), set(hit_list_dccnn)], fill=['number'])
    fig, ax = venn.venn5(labels, names=['REP', 'DC-CNN'])
    # labels = venn.get_labels([set(hit_list_rep), set(hit_list_pairs), set(hit_list_sabd), 
    # set(hit_list_dccnn), set(hit_list_hindbr)], fill=['number', 'logic'])
    # fig, ax = venn.venn5(labels, names=['0:REP', '1:Siamese Pair', '2:SABD', '3:DCCNN', '4:HINDBR'])
    # fig.show()
    fig.savefig('rep-dccnn.pdf')
    
    hit_list_rep = get_hits('rep')
    hit_list_hindbr = get_hits('hindbr')
    labels = venn.get_labels([set(hit_list_rep), set(hit_list_hindbr)], fill=['number'])
    fig, ax = venn.venn5(labels, names=['REP', 'HINDBR'])
    # labels = venn.get_labels([set(hit_list_rep), set(hit_list_pairs), set(hit_list_sabd), 
    # set(hit_list_dccnn), set(hit_list_hindbr)], fill=['number', 'logic'])
    # fig, ax = venn.venn5(labels, names=['0:REP', '1:Siamese Pair', '2:SABD', '3:DCCNN', '4:HINDBR'])
    # fig.show()
    fig.savefig('rep-hindbr.pdf')


def draw_venn_diagram():
    # the number was got by calling get_venn_diagram_numbers()
    v = venn2_unweighted(subsets = (885, 391, 5005), set_labels = ('REP',  'SABD'), set_colors=('#2f77b0', '#4a9c3e'), alpha = 0.8);
    
    for text in v.set_labels:
        text.set_fontsize(16)
    for x in range(len(v.subset_labels)):
        if v.subset_labels[x] is not None:
            v.subset_labels[x].set_fontsize(16)

    plt.savefig('f-rep-sabd.pdf')
    plt.clf()

    v = venn2_unweighted(subsets = (1628, 260, 4262), set_labels = ('REP', 'Siamese Pair'), set_colors=('#2f77b0', '#f58633'), alpha = 0.8);
    for text in v.set_labels:
        text.set_fontsize(16)
    for x in range(len(v.subset_labels)):
        if v.subset_labels[x] is not None:
            v.subset_labels[x].set_fontsize(16)
    plt.savefig('f-rep-pair.pdf')
    plt.clf()
    
    v = venn2_unweighted(subsets = (4570, 31, 1320), set_labels = ('REP', 'HINDBR'), set_colors=('#2f77b0', '#8e6cb8'), alpha = 0.8);
    for text in v.set_labels:
        text.set_fontsize(16)
    for x in range(len(v.subset_labels)):
        if v.subset_labels[x] is not None:
            v.subset_labels[x].set_fontsize(16)
    plt.savefig('f-rep-hindbr.pdf')
    plt.clf()
    
    v = venn2_unweighted(subsets = (3668, 89, 2222), set_labels = ('REP', 'DC-CNN'), set_colors=('#2f77b0', '#87594d'), alpha = 0.8);
    for text in v.set_labels:
        text.set_fontsize(16)
    for x in range(len(v.subset_labels)):
        if v.subset_labels[x] is not None:
            v.subset_labels[x].set_fontsize(16)
    plt.savefig('f-rep-dccnn.pdf')
    
if __name__ == '__main__':
    # for method in methods:
    #     calculate_all_method(method)
    # total_hits, total_hit_list = 0, [] 
    # log_files = [filename for filename in os.listdir('./drawing/') if filename.startswith("{}".format('recommend_ranknet'))]
    # for dataset in datasets:
    #     # print(dataset)
    #     for log_file in log_files:
    #         if dataset in log_file:
    #             num_hits, hit_list = extract_result_from_rep('./drawing/{}'.format(log_file), dataset)    
    #             total_hit_list.extend(hit_list)
    #             total_hits += num_hits
    # print('REP: {}'.format(total_hits))
    # with open('./drawing/rep_all_hits.txt', 'w') as f:
    #     for item in total_hit_list:
    #         f.write('{}\n'.format(item))
    # get_venn_diagram_numbers()
    # draw_venn_diagram1()
    # draw_venn_diagram2()
    # draw_venn_diagram3()
    draw_venn_diagram()