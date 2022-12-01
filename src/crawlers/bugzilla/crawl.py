import logging
import configparser
import os
import random
import socket
import time
from urllib.request import *

# Import and init common logger
import sys
sys.path.append("../")
from logger import init_logger
init_logger()

if len(sys.argv) < 2:
    print("Usage: python crawl.py [PROJECT_NAME]")
    sys.exit(0)
project_name = sys.argv[1].upper()

socket.setdefaulttimeout(50)
config = configparser.ConfigParser()
config.read("settings.ini")

if project_name not in config:
    print("No configuration for %s in settings.ini."%project_name)
    sys.exit(1)

BASE_URL = config[project_name]['base_url']
OUTPUT_DIR = config[project_name]['output_dir']
START_INDEX = int(config[project_name]['start_index'])
END_INDEX = int(config[project_name]['end_index'])

if not os.path.isdir(OUTPUT_DIR):
    logging.info("Creating directory: %s" % OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)

# Crawl latest bug report first
for i in range(START_INDEX, END_INDEX, -1):
    file_name = "%s/%s.xml" % (OUTPUT_DIR, i)
    if os.path.isfile(file_name):
        continue

    if i % 5 == 0:
        rand_time = random.randint(3, 7)
        logging.info("Sleep for %s seconds" % rand_time)
        time.sleep(rand_time)
    url = BASE_URL % i
    try:
        logging.info(url)
        resp = urlopen(url)
    except Exception as e:
        logging.error("Request Failed!", e)
        continue

    try:
        content = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        logging.error("Decoding Failed!", e)
        continue

    f = open(file_name, "w")
    if content.startswith(")]}'"):
        content = "\n".join(content.split("\n")[1:])

    f.write(content)
    f.close()

