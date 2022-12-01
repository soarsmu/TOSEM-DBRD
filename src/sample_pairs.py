"""
Created on 12 Dec 2021
sample equal number of positive & negative pairs
for training
"""

import argparse
import random
random.seed(42)

import os


def sample_training_split():
    target_pair_number = int(args.target_num)
    positive_lines, negative_lines = list(), list()
    with open(original_training_split_pair_file) as f:
        lines = f.readlines()
        
    for line in lines:
        if line.strip().split(',')[-1] == '1':
            positive_lines.append(line)
        else:
            negative_lines.append(line)
    
    sampled_pairs = random.sample(positive_lines, target_pair_number)
    sampled_pairs.extend(random.sample(negative_lines, target_pair_number))
    random.shuffle(sampled_pairs)
    
    with open(sampled_training_split_pair_file, 'w') as f:
        for line in sampled_pairs:
            f.write(line)
    print(len(sampled_pairs))
    
def sample_valid():
    target_pair_number = int(args.target_num)
    
    positive_lines, negative_lines = list(), list()
    with open(original_valid_pair_file) as f:
        lines = f.readlines()
        
    for line in lines:
        if line.strip().split(',')[-1] == '1':
            positive_lines.append(line)
        else:
            negative_lines.append(line)
            
    sampled_pairs = random.sample(positive_lines, target_pair_number)
    sampled_pairs.extend(random.sample(negative_lines, target_pair_number))
    random.shuffle(sampled_pairs)
    
    with open(sampled_valid_pair_file, 'w') as f:
        for line in sampled_pairs:
            f.write(line)
    print(len(sampled_pairs))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--project', help='project name', required=True)
    parser.add_argument('--type', help='validation or training', required=True)
    parser.add_argument('--rq', help='jira, github, age', required=True)
    parser.add_argument('--target_num', help='target_num', required=True)
    
    args = parser.parse_args()
    original_training_split_pair_file = '../SABD/dataset/{}/training_split_{}_pairs_random_1.txt'.format(args.project, args.project)
    sampled_training_split_pair_file = '../SABD/dataset/{}/sampled_{}_training_split_{}_pairs_random_1.txt'.format(args.project, args.rq, args.project)
    
    original_valid_pair_file = '../SABD/dataset/{}/validation_{}_pairs_random_1.txt'.format(args.project, args.project)
    sampled_valid_pair_file = '../SABD/dataset/{}/sampled_{}_validation_{}_pairs_random_1.txt'.format(args.project, args.rq, args.project)
    
    target_pair_number = int(args.target_num)
    
    if args.type == 'valid':
        if not os.path.exists(sampled_valid_pair_file):
            print('sampling valid')
            sample_valid()
    if args.type == 'train':
        if not os.path.exists(sampled_training_split_pair_file):
            print('sampling training_split')
            sample_training_split()