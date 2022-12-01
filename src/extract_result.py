"""
Created on 2 Aug 2021
Updated on 25 Aug 2022
"""

pair_eclipse = 'SABD/result_log/pairs_eclipse_2021-12-17-20:39:11.log'
pair_eclipse_initial = 'SABD/result_log/pairs_eclipse-initial_2021-12-17-21:10:35.log'
pair_eclipse_old = 'SABD/result_log/pairs_eclipse-old_2021-12-28-19:48:29.log'

sabd_eclipse = 'SABD/result_log/sabd_eclipse_2021-12-17-21:38:51.log'
sabd_eclipse_initial = 'SABD/result_log/sabd_eclipse-initial_2021-12-17-21:40:05.log'
sabd_eclipse_old = 'SABD/result_log/sabd_eclipse-old_2021-12-28-19:48:36.log'

pair_mozilla = '/SABD/result_log/pairs_mozilla_2021-12-18-10:03:30.log'
pair_mozilla_initial = 'SABD/result_log/pairs_mozilla-initial_2021-12-18-10:59:50.log'
pair_mozilla_old = 'SABD/result_log/pairs_mozilla-old_2021-12-18-10:04:32.log'

sabd_mozilla = 'SABD/result_log/sabd_mozilla_2021-12-18-14:27:32.log'
sabd_mozilla_initial = 'SABD/result_log/sabd_mozilla-initial_2021-12-18-21:15:51.log'
sabd_mozilla_old = 'SABD/result_log/sabd_mozilla-old_2021-12-21-23:30:03.log'

pair_spark = 'SABD/result_log/pairs_spark_2021-12-19-15:40:56.log'
pair_hadoop = 'SABD/result_log/pairs_hadoop_2021-12-19-15:41:04.log'
pair_eclipse_jira = 'SABD/result_log/pairs_jira_eclipse_2021-12-19-16:16:39.log'
pair_mozilla_jira = 'SABD/result_log/pairs_jira_mozilla_2021-12-20-13:21:05.log'

sabd_spark = 'SABD/result_log/sabd_spark_2021-12-19-15:38:57.log'
sabd_hadoop = 'SABD/result_log/sabd_hadoop_2021-12-19-15:37:12.log'
sabd_eclipse_jira = 'SABD/result_log/sabd_jira_eclipse_2021-12-19-15:50:15.log'
sabd_mozilla_jira = 'SABD/result_log/sabd_jira_mozilla_2021-12-19-16:42:25.log'

related_folder = '../'

## Without Controling size
age_sabd_eclipse_recent = 'SABD/result_log/sabd_eclipse_2021-12-17-21:38:51.log'
age_sabd_eclipse_old = 'SABD/result_log/sabd_eclipse-old_2021-12-26-15:14:23.log'
age_pair_eclipse_recent = 'SABD/result_log/pairs_eclipse_2021-12-17-20:39:11.log'
age_pair_eclipse_old = 'SABD/result_log/pairs_eclipse-old_2021-12-17-20:54:30.log'

age_sabd_mozilla_recent = 'SABD/result_log/sabd_mozilla_2021-12-20-19:47:48.log'
age_sabd_mozilla_old = 'SABD/result_log/sabd_mozilla-old_2021-12-25-17:08:25.log'
age_pair_mozilla_recent = 'SABD/result_log/pairs_mozilla_2021-12-20-19:52:34.log'
age_pair_mozilla_old = 'SABD/result_log/pairs_mozilla-old_2021-12-26-11:33:38.log'

state_sabd_eclipse_updated = 'SABD/result_log/sabd_eclipse_2021-12-17-21:38:51.log'
state_sabd_eclipse_initial = 'SABD/result_log/sabd_eclipse-initial_2021-12-26-18:34:37.log'
state_pair_eclipse_updated = 'SABD/result_log/pairs_eclipse_2021-12-17-20:39:11.log'
state_pair_eclipse_initial = 'SABD/result_log/pairs_eclipse-initial_2021-12-26-19:01:35.log'

state_sabd_mozilla_updated = 'SABD/result_log/sabd_mozilla_2021-12-20-19:47:48.log'
state_sabd_mozilla_initial = 'SABD/result_log/sabd_mozilla-initial_2021-12-26-20:05:43.log'
state_pair_mozilla_updated = 'SABD/result_log/pairs_mozilla_2021-12-20-19:52:34.log'
state_pair_mozilla_initial = 'SABD/result_log/pairs_mozilla-initial_2021-12-26-08:54:09.log'


mozilla_old_result = './log/bugzilla_full_text_search_mozilla-old.log'
eclipse_old_result = './log/bugzilla_full_text_search_eclipse-old.log'
import re


def extract_recall_list_from_rep(result_file) -> list:
    """
    extract recall rate from log files
    """
    
    with open(result_file, 'r') as f:
        lines = f.readlines()

    recall = list()

    if len(re.findall(r'Average Average Recall', lines[-22])) > 0:
        for line_index in range(-21, -1):
            recall.append(float(lines[line_index].strip()))

    for i in recall[:10]:
        print(i)


def count_inf(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()

    position_line = lines[-1]
    num_inf = 0
    for item in re.findall('\[(.*)\]',position_line)[0].split(','):
        if item.strip() == 'inf':
            num_inf += 1
            
    print('number of inf: {}'.format(num_inf))
    print('percentage is: {}'.format(num_inf / len(re.findall('\[(.*)\]',position_line)[0].split(','))))
    
def count_inf_other(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()

    position_line = lines[-3]
    num_inf = 0
    for item in re.findall('\[(.*)\]',position_line)[0].split(','):
        if item.strip() == 'inf':
            num_inf += 1
            
    print('number of inf: {}'.format(num_inf))
    print('percentage is: {}'.format(num_inf / len(re.findall('\[(.*)\]',position_line)[0].split(','))))
    
def get_recall(file):
    with open(file, 'r') as f:
        data = f.read()

    tmp_list = re.findall(r"\'rate\': 0\.\d+", data)[:10]
    
    for i, content in zip(range(len(tmp_list)), tmp_list):
        print('{}'.format(re.findall(r"0\.\d+", content)[0]))

def list_fts():
    print('eclipse')
    get_recall('./log/bugzilla_full_text_search_eclipse.log')
    
    print('mozilla')
    get_recall('./log/bugzilla_full_text_search_mozilla.log')
    
    print('hadoop')
    get_recall('./log/bugzilla_full_text_search_hadoop.log')
    
    print('spark')
    get_recall('./log/bugzilla_full_text_search_spark.log')
    
    print('kibana')
    get_recall('./log/bugzilla_full_text_search_kibana.log')
    
    print('vscode')
    get_recall('./log/bugzilla_full_text_search_vscode.log')
    
    
def list_age_eclipse():
    print('sabd recent eclipse')
    get_recall(age_sabd_eclipse_recent)
    print('sabd old eclipse')
    get_recall(age_sabd_eclipse_old)
    print('pair recent eclipse')
    get_recall(age_pair_eclipse_recent)
    print('pair old eclipse')
    get_recall(age_pair_eclipse_old)
    
def list_state_eclipse():
    print('sabd updated eclipse')
    get_recall(state_sabd_eclipse_updated)
    print('sabd initial eclipse')
    get_recall(state_sabd_eclipse_initial)
    print('pair updated eclipse')
    get_recall(state_pair_eclipse_updated)
    print('pair initial eclipse')
    get_recall(state_pair_eclipse_initial)

def list_state_mozilla():
    print('sabd updated mozilla')
    get_recall(state_sabd_mozilla_updated)
    print('\nsabd initial mozilla')
    get_recall(state_sabd_mozilla_initial)
    print('\npair updated mozilla')
    get_recall(state_pair_mozilla_updated)
    print('\npair initial mozilla')
    get_recall(state_pair_mozilla_initial)

def controlled_size_age_eclipse():
    print('sabd recent eclipse')
    get_recall(sabd_eclipse)
    print('\nsabd old eclipse')
    get_recall(sabd_eclipse_old)
    print('\npair recent eclipse')
    get_recall(pair_eclipse)
    print('\npair old eclipse')
    get_recall(pair_eclipse_old)

def controlled_size_age_mozilla():
    print('sabd recent mozilla')
    get_recall(sabd_mozilla)
    print('\nsabd old mozilla')
    get_recall(sabd_mozilla_old)
    print('\npair recent mozilla')
    get_recall(pair_mozilla)
    print('\npair old mozilla')
    get_recall(pair_mozilla_old)
    
    
def controlled_size_state_eclipse():
    print('sabd updated eclipse')
    get_recall(sabd_eclipse)
    print('\nsabd initial eclipse')
    get_recall(sabd_eclipse_initial)
    print('\npair updated eclipse')
    get_recall(pair_eclipse)
    print('\npair initial eclipse')
    get_recall(pair_eclipse_initial)

def controlled_size_state_mozilla():
    print('sabd updated mozilla')
    get_recall(sabd_mozilla)
    print('\nsabd initial mozilla')
    get_recall(sabd_mozilla_initial)
    print('\npair updated mozilla')
    get_recall(pair_mozilla)
    print('\npair initial mozilla')
    get_recall(pair_mozilla_initial)
    
def ITS_pairs():
    # get_recall(related_folder + 'SABD/result_log/pairs_eclipse_2021-12-28-16:59:47.log')
    # get_recall(related_folder + 'SABD/result_log/pairs_hadoop_2021-12-29-21:36:09.log')
    # get_recall(related_folder + 'SABD/result_log/pairs_spark_2021-12-29-21:26:47.log')
    # get_recall(related_folder + 'SABD/result_log/pairs_kibana_2021-12-29-22:12:10.log')
    print('pair')
    get_recall(related_folder + 'SABD/result_log/pairs_vscode_2021-12-29-21:45:07.log')
    
def ITS_sabd():
    # get_recall(related_folder + 'SABD/result_log/sabd_hadoop_2021-12-29-21:28:25.log')
    # get_recall(related_folder + 'SABD/result_log/sabd_spark_2021-12-29-21:26:59.log')
    # get_recall(related_folder + 'SABD/result_log/sabd_kibana_2021-12-29-21:30:03.log')
    # get_recall(related_folder + 'SABD/result_log/sabd_vscode_2021-12-29-21:30:05.log')
    get_recall(related_folder + 'SABD/result_log/sabd_kibana_2021-12-30-09:47:32.log')
    
if __name__ == '__main__':
    # print('========> sabd spark <===========')
    # get_recall(sabd_spark)
    # print()
    # print('==========> siamese spark <===========')
    # get_recall(pair_spark)
    # print()
    # print('========> sabd hadoop <===========')
    # get_recall(sabd_hadoop)
    # print()
    # print('==========> siamese hadoop <===========')
    # get_recall(pair_hadoop)
    # print('========> sabd eclipse jira <===========')
    # get_recall(sabd_eclipse_its)
    # print()
    # print('==========> siamese eclipse jira <===========')
    # get_recall(pair_eclipse_its)
    
    # print('========> sabd mozilla updated <===========')
    # get_recall(sabd_mozilla)
    # print()
    # print('==========> siamese mozilla initial <===========')
    # get_recall(sabd_mozilla_initial)
    
    # print('========> sabd mozilla jira <===========')
    # get_recall(sabd_mozilla_jira)
    # print('==========> pairs mozilla jira <===========')
    # get_recall(pair_mozilla_jira)
    # list_age_eclipse()
    # list_state_eclipse()
    # controlled_size_age_eclipse()
    # controlled_size_age_mozilla()
    # controlled_size_state_eclipse()
    # controlled_size_state_mozilla()
    # list_state_mozilla()
    # mozilla_result_folder = '/home/zhaozheng/dbrd-results/mozilla/'
    # REP_folder = '/home/zhaozheng/dbrd-results/REP/'
    ### state bias on Mozilla
    # get_recall(mozilla_result_folder + 'sabd_mozilla_2.log')
    # get_recall(mozilla_result_folder + 'pairs_mozilla_2021-08-23-12:38:30.log')
    # extract_recall_list_from_rep(REP_folder + 'dbrd_ranknet_rep_mozilla_25_Aug_5_I-1')
    
    # mozilla_initial_folder = '/home/zhaozheng/dbrd-results/mozillaInitial/'
    # get_recall(mozilla_initial_folder + 'sabd_mozillaInitial_2021-08-24-14:20:05.log')
    # get_recall(mozilla_initial_folder + 'pairs_mozillaInitial_2021-08-25-16:45:18.log')
    # extract_recall_list_from_rep(REP_folder + 'initial/dbrd_ranknet_rep_mozillaInitial_26_Aug_1_I-1')
    
    ### age bias on Mozilla
    # mozilla_old_folder = '/home/zhaozheng/dbrd-results/mozillaOld/'
    # get_recall(mozilla_old_folder + 'sabd_mozillaOld_2021-08-25-22:37:13.log')
    # get_recall(mozilla_old_folder + 'pairs_mozillaOld_2021-08-26-09:35:04.log')
    # extract_recall_list_from_rep(mozilla_old_folder + 'dbrd_ranknet_rep_mozillaOld_26_Aug_1_I-1')
    
    
    # get_recall(related_folder + 'SABD/result_log/pairs_eclipse-old_2021-12-28-19:48:29.log')
    # get_recall(related_folder + 'SABD/result_log/sabd_eclipse_2021-12-28-17:06:34.log')
    
    # get_recall(related_folder + 'SABD/result_log/pairs_eclipse-initial_2021-12-28-19:06:52.log')
    # get_recall(related_folder + 'SABD/result_log/sabd_eclipse-initial_2021-12-28-19:07:13.log')
    # get_recall(related_folder + sabd_eclipse_old)
    # get_recall(related_folder + pair_eclipse_old)
    
    # extract_recall_list_from_rep(related_folder + 'REP/dbrd_ranknet_29_Dec_eclipse_I-1')
    # extract_recall_list_from_rep(related_folder + 'REP/dbrd_ranknet_29_Dec_eclipse-initial_I-1')
    # extract_recall_list_from_rep(related_folder + 'REP/dbrd_ranknet_29_Dec_eclipse-old_I-1')
    # extract_recall_list_from_rep(related_folder + 'REP/dbrd_ranknet_29_Dec_vscode_I-1')
    # extract_recall_list_from_rep(related_folder + 'REP/dbrd_ranknet_29_Dec_kibana_I-1')
    # extract_recall_list_from_rep(related_folder + 'REP/dbrd_ranknet_29_Dec_hadoop_I-1')
    # extract_recall_list_from_rep(related_folder + 'REP/dbrd_ranknet_29_Dec_spark_I-1')
    # ITS_sabd()
    ### sampled data [eclipse]
    # get_recall(related_folder + 'SABD/result_log/sabd_sampled-eclipse_2021-12-30-15:54:40.log')
    # get_recall(related_folder + 'SABD/result_log/pairs_sampled-eclipse_2021-12-30-15:56:31.log')
    # extract_recall_list_from_rep(related_folder + 'REP/dbrd_ranknet_30_Dec_sampled-eclipse_I-1')
    
    ### [sampled-old]
    # extract_recall_list_from_rep(related_folder + 'REP/dbrd_ranknet_30_Dec_sampled-eclipse-old_I-1')
    # get_recall(related_folder + 'SABD/result_log/sabd_sampled-eclipse-old_2021-12-30-16:51:15.log')
    # get_recall(related_folder + 'SABD/result_log/pairs_sampled-eclipse-old_2021-12-30-16:52:27.log')
    
    ### [sample-eclipse-initial]
    # extract_recall_list_from_rep(related_folder + 'REP/dbrd_ranknet_30_Dec_sampled-eclipse-initial_I-1')
    # get_recall(related_folder + 'SABD/result_log/sabd_sampled-eclipse-initial_2021-12-30-17:04:49.log')
    # get_recall(related_folder + 'SABD/result_log/pairs_sampled-eclipse-initial_2021-12-30-17:05:05.log')
    
    ### [sampled mozilla]
    # extract_recall_list_from_rep(related_folder + 'REP/dbrd_ranknet_30_Dec_sampled-mozilla_I-1')
    # get_recall(related_folder + 'SABD/result_log/pairs_sampled-mozilla_2021-12-30-20:20:32.log')
    # get_recall(related_folder + 'SABD/result_log/sabd_sampled-mozilla_2021-12-30-20:05:51.log')
    
    ### [mozilla-initial]
    # extract_recall_list_from_rep(related_folder + 'REP/dbrd_ranknet_30_Dec_sampled-mozilla-initial_I-1')
    # get_recall(related_folder + 'SABD/result_log/pairs_sampled-mozilla-initial_2021-12-30-20:11:25.log')
    # get_recall(related_folder + 'SABD/result_log/sabd_sampled-mozilla-initial_2021-12-30-20:01:56.log')
    
    ### [mozilla-old]
    # extract_recall_list_from_rep(related_folder + 'REP/dbrd_ranknet_30_Dec_sampled-mozilla-old_I-1')
    # get_recall(related_folder + 'SABD/result_log/pairs_sampled-mozilla-old_2021-12-30-20:03:13.log')
    # get_recall('../SABD/result_log/sabd_sampled-mozilla-old_2021-12-30-20:06:56.log')
    
    ## four
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_30_Dec_sampled-hadoop_I-1')
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_30_Dec_sampled-spark_I-1')
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_30_Dec_sampled-kibana_I-1')
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_30_Dec_sampled-vscode_I-1')
    
    ## sabd four
    # get_recall('../SABD/result_log/sabd_sampled-hadoop_2021-12-30-21:19:40.log')
    # get_recall('../SABD/result_log/sabd_sampled-spark_2021-12-30-21:20:20.log')
    # get_recall('../SABD/result_log/sabd_sampled-kibana_2021-12-30-21:21:31.log')
    # get_recall('../SABD/result_log/sabd_sampled-vscode_2021-12-30-21:22:38.log')
    
    ## pair four
    # get_recall('../SABD/result_log/pairs_sampled-hadoop_2021-12-30-21:19:46.log')
    # get_recall('../SABD/result_log/pairs_sampled-spark_2021-12-30-21:28:58.log')
    # get_recall('../SABD/result_log/pairs_sampled-kibana_2021-12-30-21:37:27.log')
    # get_recall('../SABD/result_log/pairs_sampled-vscode_2021-12-30-21:45:49.log')
    
    # get_recall('../SABD/result_log/pairs_github_eclipse_2022-01-04-10:15:58.log')
    # get_recall('../SABD/result_log/pairs_github_mozilla_2022-01-04-10:25:15.log')
    # get_recall('../SABD/result_log/pairs_github_vscode_2022-01-04-10:01:30.log')
    # get_recall('../SABD/result_log/sabd_github_vscode_2022-01-03-22:42:11.log')
    # get_recall('../SABD/result_log/sabd_github_eclipse_2022-01-04-00:14:13.log')
    # get_recall('../SABD/result_log/pairs_github_kibana_2022-01-04-14:26:29.log')
    # get_recall('../SABD/result_log/sabd_github_kibana_2022-01-04-14:35:07.log')
    # get_recall('../SABD/result_log/sabd_mozilla-old_2021-12-21-23:30:03.log')
    # get_recall('../SABD/result_log/sabd_mozilla_2021-12-18-14:27:32.log')
    # get_recall('../SABD/result_log/sabd_github_mozilla_2022-01-04-00:29:34.log')
    ## dc-cnn
    # get_recall('../DC-CNN/result_log/eclipse_2021-12-29-16:03:50.log')
    # get_recall('../DC-CNN/result_log/hadoop_2021-12-29-15:50:18.log')
    # get_recall('../DC-CNN/result_log/spark_2021-12-29-16:07:16.log')
    # get_recall('../DC-CNN/result_log/kibana_2022-01-04-17:03:06.log')
    # get_recall('../DC-CNN/result_log/vscode_2022-01-04-17:11:43.log')
    # get_recall('../DC-CNN/result_log/mozilla_2022-01-04-20:27:47.log')
    
    ## hindbr
    # get_recall('../HINDBR/result_log/issre_eclipse_TEXT_2021-12-29-12:58:52.log')
    # get_recall('../HINDBR/result_log/issre_mozilla_TEXT_2022-01-01-22:15:00.log')
    # get_recall('../HINDBR/result_log/issre_hadoop_TEXT_2021-12-29-14:00:01.log')
    # get_recall('../HINDBR/result_log/issre_spark_TEXT_2021-12-29-14:05:56.log')
    # get_recall('../HINDBR/result_log/issre_vscode_TEXT_2021-12-31-20:38:42.log')
    # get_recall('../HINDBR/result_log/issre_kibana_TEXT_2022-01-05-16:02:04.log')
    
    # list_fts()
    # ITS_pairs()
    # count_inf(mozilla_old_result)
    # count_inf(eclipse_old_result)
    # count_inf('../SABD/result_log/pairs_eclipse_2021-12-28-16:59:47.log')
    # count_inf('../SABD/result_log/pairs_eclipse-old_2021-12-28-19:48:29.log')
    # count_inf_other('../SABD/result_log/sabd_eclipse-old_2021-12-28-19:48:36.log')
    
    ### Hadoop-1day sabd
    # get_recall('../SABD/result_log/sabd_hadoop-1day_2022-08-16-10:11:56.log')
    # get_recall('../SABD/result_log/sabd_hadoop-1day_2022-08-16-10:13:28.log')
    # get_recall('../SABD/result_log/sabd_hadoop-1day_2022-08-16-10:14:59.log')
    # get_recall('../SABD/result_log/sabd_hadoop-1day_2022-08-16-10:16:29.log')
    # get_recall('../SABD/result_log/sabd_hadoop-1day_2022-08-16-10:18:01.log')
    
    ### hadoop sabd
    # get_recall('../SABD/result_log/sabd_hadoop_2022-08-16-11:13:43.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_hadoop_2022-08-16-11:15:14.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_hadoop_2022-08-16-11:16:46.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_hadoop_2022-08-16-11:18:18.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_hadoop_2022-08-16-11:19:55.log')
    
    ### Spark-1day sabd
    # get_recall('../SABD/result_log/sabd_spark-1day_2022-08-16-10:49:05.log')
    # get_recall('../SABD/result_log/sabd_spark-1day_2022-08-16-10:50:24.log')
    # get_recall('../SABD/result_log/sabd_spark-1day_2022-08-16-10:51:43.log')
    # get_recall('../SABD/result_log/sabd_spark-1day_2022-08-16-10:53:03.log')
    # get_recall('../SABD/result_log/sabd_spark-1day_2022-08-16-10:54:21.log')
    
    ### spark
    # get_recall('../SABD/result_log/sabd_spark_2022-08-16-11:07:47.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_spark_2022-08-16-13:10:13.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_spark_2022-08-16-11:09:41.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_spark_2022-08-16-11:11:02.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_spark_2022-08-16-11:12:25.log')
    
    ### pair hadoop-1day
    # get_recall('../SABD/result_log/pairs_hadoop-1day_2022-08-16-10:14:55.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_hadoop-1day_2022-08-16-10:23:14.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_hadoop-1day_2022-08-16-10:31:28.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_hadoop-1day_2022-08-16-10:39:47.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_hadoop-1day_2022-08-16-10:47:56.log')
    
    ### hadoop pair
    # get_recall('../SABD/result_log/pairs_hadoop_2022-08-16-11:10:41.log')
    # print('--'*10)
    # get_recall('../SABD/result_log/pairs_hadoop_2022-08-16-11:20:03.log')
    # print('--'*10)
    # get_recall('../SABD/result_log/pairs_hadoop_2022-08-16-11:28:38.log')
    # print('--'*10)
    # get_recall('../SABD/result_log/pairs_hadoop_2022-08-16-11:36:56.log')
    # print('--'*10)
    # get_recall('../SABD/result_log/pairs_hadoop_2022-08-16-11:45:20.log')
    
    ### pair spark
    # get_recall('../SABD/result_log/pairs_spark_2022-08-16-11:09:19.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_spark_2022-08-16-11:19:17.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_spark_2022-08-16-11:28:07.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_spark_2022-08-16-11:36:38.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_spark_2022-08-16-11:45:01.log')
    
    ### pair spark-1day
    # get_recall('../SABD/result_log/pairs_spark-1day_2022-08-16-11:03:03.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_spark-1day_2022-08-16-11:11:13.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_spark-1day_2022-08-16-11:20:42.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_spark-1day_2022-08-16-11:29:21.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_spark-1day_2022-08-16-11:37:33.log')
    
    ### pair eclipse
    # get_recall('../SABD/result_log/pairs_eclipse_2022-08-17-14:40:32.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_eclipse_2022-08-17-14:52:23.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_eclipse_2022-08-17-15:04:42.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_eclipse_2022-08-17-15:17:24.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_eclipse_2022-08-17-15:30:01.log')

    ### Pair eclipse-old
    # get_recall('../SABD/result_log/pairs_eclipse-old_2022-08-17-15:42:31.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_eclipse-old_2022-08-17-16:14:47.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_eclipse-old_2022-08-17-16:47:04.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_eclipse-old_2022-08-17-17:19:36.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_eclipse-old_2022-08-17-17:50:26.log')

    ### Pair eclipse-initial
    # get_recall('../SABD/result_log/pairs_eclipse-initial_2022-08-17-16:12:52.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_eclipse-initial_2022-08-17-16:24:42.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_eclipse-initial_2022-08-17-16:40:43.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_eclipse-initial_2022-08-17-16:53:55.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_eclipse-initial_2022-08-17-17:08:27.log')

    ### SABD eclipse
    # get_recall('../SABD/result_log/sabd_eclipse_2022-08-17-14:39:15.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_eclipse_2022-08-17-14:54:12.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_eclipse_2022-08-17-15:09:20.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_eclipse_2022-08-17-15:24:31.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_eclipse_2022-08-17-15:40:04.log')

    ### SABD eclipse old
    # get_recall('../SABD/result_log/sabd_eclipse-old_2022-08-17-15:55:39.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_eclipse-old_2022-08-17-20:52:18.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_eclipse-old_2022-08-18-01:55:05.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_eclipse-old_2022-08-18-06:49:25.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_eclipse-old_2022-08-18-11:50:05.log')

    ### SABD eclipse initial
    # get_recall('../SABD/result_log/sabd_eclipse-initial_2022-08-17-14:55:15.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_eclipse-initial_2022-08-17-15:10:12.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_eclipse-initial_2022-08-17-15:25:11.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_eclipse-initial_2022-08-17-15:40:15.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_eclipse-initial_2022-08-17-15:55:42.log')

    ### SABD kibana
    # get_recall('../SABD/result_log/sabd_kibana_2022-08-17-19:56:51.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_kibana_2022-08-17-20:01:31.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_kibana_2022-08-17-20:06:11.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_kibana_2022-08-17-20:10:57.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_kibana_2022-08-17-20:15:43.log')

    ### SABD vscode
    # get_recall('../SABD/result_log/sabd_vscode_2022-08-17-19:57:28.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_vscode_2022-08-17-21:32:47.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_vscode_2022-08-17-23:11:18.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_vscode_2022-08-18-00:46:15.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_vscode_2022-08-18-02:15:35.log')
    
    ### SABD mozilla
    # get_recall('../SABD/result_log/sabd_mozilla_2022-08-17-14:46:57.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_mozilla_2022-08-18-14:00:20.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_mozilla_2022-08-19-13:02:55.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_mozilla_2022-08-20-12:08:48.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_mozilla_2022-08-21-11:05:15.log')
    
    ### SABD mozilla-initial
    # get_recall('../SABD/result_log/sabd_mozilla-initial_2022-08-17-14:57:04.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_mozilla-initial_2022-08-18-14:05:04.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_mozilla-initial_2022-08-19-13:02:13.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_mozilla-initial_2022-08-20-12:03:17.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_mozilla-initial_2022-08-21-11:03:00.log')
    
    ### SABD old
    # get_recall('../SABD/result_log/sabd_mozilla-old_2022-08-19-14:40:33.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_mozilla-old_2022-08-19-14:44:24.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_mozilla-old_2022-08-19-19:52:49.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_mozilla-old_2022-08-19-20:15:55.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_mozilla-old_2022-08-22-10:19:52.log')
    
    ### Pair kibana
    # get_recall('../SABD/result_log/pairs_kibana_2022-08-17-20:20:28.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_kibana_2022-08-17-20:31:27.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_kibana_2022-08-17-20:42:26.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_kibana_2022-08-17-20:55:54.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_kibana_2022-08-17-21:09:21.log')

    ### Pair vscode
    # get_recall('../SABD/result_log/pairs_vscode_2022-08-18-03:44:46.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_vscode_2022-08-18-04:12:54.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_vscode_2022-08-18-04:45:21.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_vscode_2022-08-18-05:17:26.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_vscode_2022-08-18-05:50:05.log')

    ### Pair mozilla
    # get_recall('../SABD/result_log/pairs_mozilla_2022-08-17-14:46:32.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_mozilla_2022-08-17-17:11:11.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_mozilla_2022-08-17-19:41:49.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_mozilla_2022-08-17-22:43:23.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_mozilla_2022-08-18-01:11:54.log')

    ### Pair Mozilla-initial
    # get_recall('../SABD/result_log/RQ1-State-Bugzilla/pairs_mozilla-initial_2022-08-20-16:46:47.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/RQ1-State-Bugzilla/pairs_mozilla-initial_2022-08-20-19:53:46.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/RQ1-State-Bugzilla/pairs_mozilla-initial_2022-08-20-22:43:33.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/RQ1-State-Bugzilla/pairs_mozilla-initial_2022-08-21-01:50:08.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/RQ1-State-Bugzilla/pairs_mozilla-initial_2022-08-21-04:39:11.log')

    ### Pair Mozilla-old
    # get_recall('../SABD/result_log/pairs_mozilla-old_2022-08-21-07:47:03.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_mozilla-old_2022-08-21-14:19:49.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_mozilla-old_2022-08-22-00:33:16.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_mozilla-old_2022-08-22-07:44:53.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_mozilla-old_2022-08-22-16:36:11.log')

    ### Pair Hadoop-old
    # get_recall('../SABD/result_log/pairs_hadoop-old_2022-08-18-14:08:13.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_hadoop-old_2022-08-18-14:26:03.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_hadoop-old_2022-08-18-14:43:31.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_hadoop-old_2022-08-18-15:01:54.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/pairs_hadoop-old_2022-08-18-15:20:10.log')

    ### SABD hadoop-old
    # get_recall('../SABD/result_log/sabd_hadoop-old_2022-08-18-14:00:52.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_hadoop-old_2022-08-18-14:15:31.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_hadoop-old_2022-08-18-14:32:24.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_hadoop-old_2022-08-18-14:49:20.log')
    # print('---'*10)
    # get_recall('../SABD/result_log/sabd_hadoop-old_2022-08-18-15:05:27.log')
    
    ### REP eclipse
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_eclipse-1_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_eclipse-2_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_eclipse-3_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_eclipse-4_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_eclipse-5_I-1')
    
    ### REP eclise-initial
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_eclipse-initial-1_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_eclipse-initial-2_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_eclipse-initial-3_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_eclipse-initial-4_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_eclipse-initial-5_I-1')
    
    ### REP eclipse-old
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_eclipse-old-1_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_eclipse-old-2_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_eclipse-old-3_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_eclipse-old-4_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_eclipse-old-5_I-1')
    
    ### REP kibana
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_kibana-1_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_kibana-2_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_kibana-3_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_kibana-4_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_kibana-5_I-1')
    
    ### REP vscode
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_vscode-1_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_vscode-2_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_vscode-3_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_vscode-4_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_vscode-5_I-1')
    
    ### REP mozilla
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_mozilla-1_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_mozilla-2_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_mozilla-3_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_mozilla-4_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_mozilla-5_I-1')
    
    ### REP mozilla-initial
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_mozilla-initial-1_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_mozilla-initial-2_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_mozilla-initial-3_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_mozilla-initial-4_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_mozilla-initial-5_I-1')
    
    ### REP mozilla-old
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_mozilla-old-1_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_mozilla-old-2_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_mozilla-old-3_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_mozilla-old-4_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_mozilla-old-5_I-1')
    
    ### REP hadoop-old
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_hadoop-old-1_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_hadoop-old-2_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_hadoop-old-3_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_hadoop-old-4_I-1')
    # print('---'*10)
    # extract_recall_list_from_rep('../REP/dbrd_ranknet_22_Aug_2022_hadoop-old-5_I-1')