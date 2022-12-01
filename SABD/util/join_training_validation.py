"""
Example:

python util/join_training_validation.py -ts /YOUR_HOME_DIRECTORY/approaches/msr/dataset/eclipse/training_split_eclipse.txt -tsp /YOUR_HOME_DIRECTORY/approaches/msr/dataset/eclipse/training_split_eclipse_pairs_random_1.txt -v /YOUR_HOME_DIRECTORY/approaches/msr/dataset/eclipse/validation_eclipse.txt -vp /YOUR_HOME_DIRECTORY/approaches/msr/dataset/eclipse/validation_eclipse_pairs_random_1.txt -t /YOUR_HOME_DIRECTORY/approaches/msr/dataset/eclipse/training_eclipse.txt  -tp dataset/eclipse/training_eclipse_pairs_random_1.txt -tst /YOUR_HOME_DIRECTORY/approaches/msr/dataset/eclipse/training_split_eclipse_triplets_random_1.txt -vt /YOUR_HOME_DIRECTORY/approaches/msr/dataset/eclipse/validation_eclipse_triplets_random_1.txt -tt /YOUR_HOME_DIRECTORY/approaches/msr/dataset/eclipse/training_eclipse_triplets_random_1.txt
"""

import argparse
import codecs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-ts', required=True, help="")
    parser.add_argument('-v', required=True, help="")
    parser.add_argument('-tsp', required=True, help="")
    parser.add_argument('-vp', required=True, help="")
    # parser.add_argument('-tst', required=True, help="")
    # parser.add_argument('-vt', required=True, help="")

    parser.add_argument('-t', required=True, help="")
    parser.add_argument('-tp', required=True, help="")
    # parser.add_argument('-tt', required=True, help="")

    args = parser.parse_args()

    training_file = codecs.open(args.t, 'w')
    training_pairs_file = codecs.open(args.tp, 'w')
    # training_triplets_file = codecs.open(args.tt, 'w')

    training_split_file = codecs.open(args.ts,'r')
    training_split_pairs_file = codecs.open(args.tsp, 'r')
    # training_split_triplets_file = codecs.open(args.tst, 'r')

    validation_file = codecs.open(args.v, 'r')
    validation_pairs_file = codecs.open(args.vp, 'r')
    # validation_triplets_file = codecs.open(args.vt, 'r')

    for l in training_split_pairs_file:
        training_pairs_file.write(l)

    for l in validation_pairs_file:
        training_pairs_file.write(l)

    # for l in training_split_triplets_file:
    #     training_triplets_file.write(l)

    # for l in validation_triplets_file:
    #     training_triplets_file.write(l)

    training_file.write(training_split_file.readline().rstrip())
    training_file.write(" ")
    training_file.write(validation_file.readline().rstrip())
    training_file.write("\n")

    training_file.write(training_split_file.readline().rstrip())
    training_file.write(" ")
    training_file.write(validation_file.readline().rstrip())
    training_file.write("\n")

    training_file.write(training_split_file.readline().rstrip())
    training_file.write(" ")
    training_file.write(validation_file.readline().rstrip())
    training_file.write("\n")