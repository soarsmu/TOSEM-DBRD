"""
Merge two files created by glove

COMMAND: 
python util/merge_wv_files.py --main word_embedding/glove.42B.300d.txt --aux word_embedding/mozilla/glove_300d_mozilla.txt --output word_embedding/mozilla/glove_42B_300d_mozilla_soft.txt
"""

import argparse

parser = argparse.ArgumentParser()

from tqdm import tqdm
# Global arguments
parser.add_argument('--main', required=True, help="Main glove file")
parser.add_argument('--aux', required=True, help="Additional word embeddings")
parser.add_argument('--output', required=True, help="")

args = parser.parse_args()

outFile = open(args.output, 'w')
mainFile = open(args.main, 'r')

wordSet = set()

for l in tqdm(mainFile):
    word = l.split(' ')[0]
    wordSet.add(word)

    outFile.write(l)

outFile.write('\n')
mainFile.close()

auxFile = open(args.aux, 'r')

for idx, l in enumerate(tqdm(auxFile)):
    word = l.split(' ')[0]

    if word in wordSet:
        continue

    outFile.write(l)
