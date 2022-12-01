from modules import generate_bug_pairs
import random, os, logging
import argparse
from tqdm import tqdm
random.seed(42)

''' Generage model training data for training and test HINDBR'''

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--project', help='project name', required=True)

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info('loading...')

logging.info("Generating model training samples for project: " + args.project)

BUG_GROUP = 'data/bug_report_groups/' + args.project + '_all.pkl'
SAVE_FILE_NAME = 'data/model_training/' + args.project + '_all.csv'

if os.path.isfile(SAVE_FILE_NAME):
    os.remove(SAVE_FILE_NAME)

# Generate duplicate pairs and non-duplicate pairs
bug_pairs = generate_bug_pairs(BUG_GROUP)
duplicate_pairs = bug_pairs[0]
non_duplicate_pairs = bug_pairs[1]

# Generate labels
dup_bug_pairs_with_label = [pair + ('1',) for pair in duplicate_pairs]
non_dup_bug_pairs_with_label = [pair + ('0',) for pair in non_duplicate_pairs]

if args.project == 'eclipse':
    # Eclipse 
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 1))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)
elif args.project == 'freedesktop':
    # Freedesktop
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 0.3))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)
elif args.project == 'gcc':
    # GCC
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 0.2))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)
elif args.project == 'gnome':
    # GNOME
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 0.005))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)
elif args.project == 'kde':
    # KDE
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 0.03))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)
elif args.project == 'libreoffice':
    # LibreOffice
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 0.2))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)
elif args.project == 'linux':
    # Linux kernel
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 1))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)
elif args.project == 'llvm':
    # LLVM
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 1))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)
elif args.project == 'openoffice':
    # OpenOffice
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 0.15))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)


bug_pairs_with_label = dup_bug_pairs_with_label + non_dup_bug_pairs_with_label
random.shuffle(bug_pairs_with_label)


logging.info('Total pairs: ' + str(len(bug_pairs_with_label)))
with open('data/model_training/{}_training_pairs.txt'.format(args.project), 'w') as f:
    for pair in tqdm(bug_pairs_with_label):
        f.write(str(pair[0]) + ',' + str(pair[1]) + ',' + pair[2])
        f.write('\n')
logging.info("Done!")