import re

def get_recall(file):
    with open(file, 'r') as f:
        data = f.read()

    tmp_list = re.findall(r"\'rate\': 0\.\d+", data)[:10]
    
    for i, content in zip(range(len(tmp_list)), tmp_list):
        print('{}'.format(re.findall(r"0\.\d+", content)[0]))

def get_run_time(file):
    with open(file, 'r') as f:
        data = f.read()

    tmp_list = re.findall(r"seconds=\d+\.\d+", data)[0]
    print('{}'.format(re.findall(r"\d+\.\d+", tmp_list)[0]))

if __name__ == '__main__':
    print('eclipse')
    # get_recall('result_log/hindbr_eclipse_TEXT_2022-08-23-10:03:07.log')
    get_run_time('result_log/hindbr_eclipse_TEXT_2022-08-23-09:28:22.log')
    get_run_time('result_log/hindbr_eclipse_TEXT_2022-08-23-09:39:56.log')
    get_run_time('result_log/hindbr_eclipse_TEXT_2022-08-23-09:51:17.log')
    get_run_time('result_log/hindbr_eclipse_TEXT_2022-08-23-10:03:07.log')
    get_run_time('result_log/hindbr_eclipse_TEXT_2022-08-26-15:42:11.log')
    print('...' * 10)

    print('mozilla')
    # get_recall('result_log/hindbr_mozilla_TEXT_2022-08-24-18:22:52.log')
    get_run_time('result_log/hindbr_mozilla_TEXT_2022-08-23-18:10:23.log')
    get_run_time('result_log/hindbr_mozilla_TEXT_2022-08-23-22:21:55.log')
    get_run_time('result_log/hindbr_mozilla_TEXT_2022-08-24-04:22:30.log')
    get_run_time('result_log/hindbr_mozilla_TEXT_2022-08-24-08:25:09.log')
    get_run_time('result_log/hindbr_mozilla_TEXT_2022-08-24-18:22:52.log')
    print('...' * 10)

    print('hadoop')
    # get_recall('result_log/hindbr_hadoop_TEXT_2022-08-30-17:07:48.log')
    get_run_time('result_log/hindbr_hadoop_TEXT_2022-08-30-17:07:48.log')
    get_run_time('result_log/hindbr_hadoop_TEXT_2022-09-03-09:56:31.log')
    get_run_time('result_log/hindbr_hadoop_TEXT_2022-09-03-09:57:04.log')
    get_run_time('result_log/hindbr_hadoop_TEXT_2022-09-03-09:57:38.log')
    get_run_time('result_log/hindbr_hadoop_TEXT_2022-09-03-09:58:12.log')
    print('...' * 10)
    
    print('spark')
    # get_recall('result_log/hindbr_spark_TEXT_2022-08-23-20:42:57.log')
    get_run_time('result_log/hindbr_spark_TEXT_2022-08-23-20:36:23.log')
    get_run_time('result_log/hindbr_spark_TEXT_2022-08-23-20:38:01.log')
    get_run_time('result_log/hindbr_spark_TEXT_2022-08-23-20:39:40.log')
    get_run_time('result_log/hindbr_spark_TEXT_2022-08-23-20:41:19.log')
    get_run_time('result_log/hindbr_spark_TEXT_2022-08-23-20:42:57.log')
    print('...' * 10)
    
    print('kibana')
    # get_recall('result_log/hindbr_kibana_TEXT_2022-08-30-17:04:40.log')
    get_run_time('result_log/hindbr_kibana_TEXT_2022-08-30-17:04:40.log')
    get_run_time('result_log/hindbr_kibana_TEXT_2022-09-03-09:57:44.log')
    get_run_time('result_log/hindbr_kibana_TEXT_2022-09-03-09:59:59.log')
    get_run_time('result_log/hindbr_kibana_TEXT_2022-09-03-10:02:12.log')
    get_run_time('result_log/hindbr_kibana_TEXT_2022-09-03-10:04:24.log')
    print('...' * 10)
    
    print('vscode')
    # get_recall('result_log/hindbr_vscode_TEXT_2022-08-23-18:11:33.log')
    get_run_time('result_log/hindbr_vscode_TEXT_2022-08-23-13:23:02.log')
    get_run_time('result_log/hindbr_vscode_TEXT_2022-08-23-14:37:16.log')
    get_run_time('result_log/hindbr_vscode_TEXT_2022-08-23-15:51:35.log')
    get_run_time('result_log/hindbr_vscode_TEXT_2022-08-23-17:00:55.log')
    get_run_time('result_log/hindbr_vscode_TEXT_2022-08-23-18:11:33.log')
    print('...' * 10)