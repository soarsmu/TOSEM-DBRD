"""
Created on 23 July 2021
"""

import logging
import os
from tqdm import tqdm

def get_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s', datefmt = '%F %A %T'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
    
    return logger


def change_filenames(path):
    project_name = os.path.split(path)[1]
    for file_name in tqdm(os.listdir(path)):
        os.rename(os.path.join(path, file_name), os.path.join(path, project_name + '_' + file_name))


def change_hadoop_file_names(project):
    project_path = '../SABD/dataset/jira/{}-xml/'.format(project)
    pre = '{}_'.format(project)

    files = os.listdir(project_path)

    for index, file in enumerate(files):
        old_name = os.path.join(project_path, file)
        suffix = os.path.basename(old_name).split('.')[0]
        os.rename(old_name, os.path.join(project_path, ''.join([pre + suffix, '.xml'])))

def move_files(path):
    # project_name = os.path.split(path)[1]
    for file_name in tqdm(os.listdir(path)):
        os.rename(os.path.join(path, file_name), os.path.join('../../our_data/spark/' + file_name))


if __name__ == '__main__':
    for PROJECT in ['HADOOP', 'HBASE', 'HDFS', 'HDT', 'HIVE', 'MAPREDUCE', 'PIG', 'YARN']:
        change_hadoop_file_names(PROJECT)