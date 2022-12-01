import os
from utils import get_logger
from datetime import datetime
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter project name.')
    parser.add_argument('--project', help='project name')

    args = parser.parse_args()

    logger = get_logger('./log/hin2vec_training_{}_{}.log'.format(args.project, datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))
    '''Train the hin embeddings on bug report hin '''
    
    if not os.path.exists('data/bug_report_hin'):
        os.makedirs('data/bug_report_hin')
    
    HIN_PATH = os.path.join(os.getcwd(), 'data/bug_report_hin/') + args.project + '.hin'
    DIM_d = 128
    NEGATIVE_SAMPLE_RATE_n = 5
    WINDOW_w = 4
    WALK_LENGTH_l = 1280
    NUM_PROCESSES = 4
    
    if not os.path.exists('data/pretrained_embeddings/hin2vec/'):
        os.makedirs('data/pretrained_embeddings/hin2vec/')
        
    NODE_VEC_SAVE_FILE = os.path.join(os.getcwd(), 'data/pretrained_embeddings/hin2vec/') + \
        args.project + "_node_" + str(DIM_d) + "d_" + str(NEGATIVE_SAMPLE_RATE_n) + "n_" + \
            str(WINDOW_w) + "w_" + str(WALK_LENGTH_l) + "l.vec"

    METAPATH_VEC_SAVE_FILE = os.path.join(os.getcwd(), 'data/pretrained_embeddings/hin2vec/') + \
        args.project + "_metapath_" + str(DIM_d) + "d_" + str(NEGATIVE_SAMPLE_RATE_n) + "n_" + \
            str(WINDOW_w) + "w_" + str(WALK_LENGTH_l) + "l.vec"

    start_time = datetime.now()
    logger.info('It started at: %s' % start_time)

    os.system("cd hin2vec/model_c/src/; make")

    #Note that hin2vec uses python2 environment.
    os.system("cd hin2vec; python2 main.py %s %s %s -d %d -n %d -w %d -l %d -p %d" \
        % (HIN_PATH,NODE_VEC_SAVE_FILE,METAPATH_VEC_SAVE_FILE,DIM_d,NEGATIVE_SAMPLE_RATE_n,WINDOW_w,WALK_LENGTH_l,NUM_PROCESSES))

    end_time = datetime.now()
    logger.info('Completed after: {}'.format(end_time - start_time))