import logging
import configparser
import os
import random
import time
import socket
from urllib.request import *

# Import and init common logger
import sys
sys.path.append("../")
from logger import init_logger
init_logger()

socket.setdefaulttimeout(50)
config = configparser.ConfigParser()
config.read("./settings.ini")

BASE_URL_TEMPLATE = config["APACHE"]["xml_base_url"]

# The project list is from Xie et al. APSEC 2018
# Detecting Duplicate Bug Reports with Convolutional Neural Networks
# The last index of each project that created in May 2021
#PREFIX_INDEX_DICT = {"HADOOP": 17737, "HDFS": 16051, "MAPREDUCE": 7348, "SPARK": 35581, "YARN": 10796}
PREFIX_INDEX_DICT = {"YARN": 10796}

for prefix, start_index in PREFIX_INDEX_DICT.items():
    output_dir = './%s-xml' % prefix
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for i in range(start_index, 0, -1):
        output_path = '%s/%s.xml' % (output_dir, i)
        if os.path.isfile(output_path):
            continue

        if i % 10 == 0:
            rand_time = random.randint(3,7)
            logging.info("Sleep for %s seconds" % rand_time)
            time.sleep(rand_time)

        bug_id = "%s-%s" % (prefix, i)
        url = BASE_URL_TEMPLATE % (bug_id, bug_id)
        logging.info(url)
        try:
            response = urlopen(url)
        except Exception as e:
            logging.error(e, exc_info=True)
            continue

        try:
            content = response.read().decode("utf-8", errors="ignore")
        except Exception as e:
            logging.error(e, exc_info=True)
            continue

        try:
            with open(output_path, "w") as f:
                f.write(content)
        except Exception as e:
            logging.error(e, exc_info=True)
            continue

