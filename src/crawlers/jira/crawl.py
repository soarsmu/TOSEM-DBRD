import logging
import configparser
import os
import socket
import json
import requests
from requests.auth import HTTPBasicAuth
from getpass import getpass

# Import and init common logger
import sys
sys.path.append("../")
from logger import init_logger
init_logger()

socket.setdefaulttimeout(50)
config = configparser.ConfigParser()
config.read("./settings.ini")

BASE_URL_TEMPLATE = config["APACHE"]["base_url"]
# The number of bug reports will be crawled
NUM_BUG_REPORTS = int(config["APACHE"]["num_bug_reports"])

# The project list is from Xie et al. APSEC 2018
# Detecting Duplicate Bug Reports with Convolutional Neural Networks
# The last index of each project that created in May 2021
PREFIX_INDEX_DICT = {"HADOOP": 17737, "HDFS": 16051, "MAPREDUCE": 7348, "SPARK": 35581}

USERNAME = input("Username: ")
PASSWORD = getpass("Password: ")

auth = HTTPBasicAuth(USERNAME, PASSWORD)
headers = {"Accept": "application/json"}
for prefix, start_index in PREFIX_INDEX_DICT.items():
    output_dir = './' + prefix
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for i in range(start_index, start_index - NUM_BUG_REPORTS, -1):
        output_path = '%s/%s.json' % (output_dir, i)

        if os.path.isfile(output_path):
            logging.info("%s is already crawled. Skip the file" % output_path)
            continue

        url = BASE_URL_TEMPLATE % (prefix, i)
        logging.info(url)
        try:
            response = requests.request("GET", url, headers=headers, auth=auth)
        except Exception as e:
            logging.error("Request Failed", e)
            continue

        try:
            with open(output_path, "w") as f:
                json_str = json.dumps(response.json())
                f.write(json_str)
        except Exception as e:
            logging.error("File save failed", e)
            continue

        response.close()

