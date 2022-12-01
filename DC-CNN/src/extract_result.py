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
    ### Eclipse
    # get_recall('../result_log/eclipse_2022-08-18-23:02:31.log')
    # print('---'*10)
    # get_recall('../result_log/eclipse_2022-08-18-23:26:25.log')
    # print('---'*10)
    # get_recall('../result_log/eclipse_2022-08-18-23:47:47.log')
    # print('---'*10)
    # get_recall('../result_log/eclipse_2022-08-19-00:09:01.log')
    # print('---'*10)
    # get_recall('../result_log/eclipse_2022-08-19-00:30:13.log')
    
    ### Mozilla
    # get_recall('../result_log/mozilla_2022-08-19-01:50:28.log')
    # print('---'*10)
    # get_recall('../result_log/mozilla_2022-08-20-09:45:36.log')
    # print('---'*10)
    # get_recall('../result_log/mozilla_2022-08-21-15:35:19.log')
    # print('--'*10)
    # get_recall('../result_log/mozilla_2022-08-22-16:16:37.log')
    # print('--'*10)
    # get_recall('../result_log/mozilla_2022-08-22-21:35:15.log')
    
    ### kibana
    # get_recall('../result_log/kibana_2022-08-19-00:53:49.log')
    # print('---'*10)
    # get_recall('../result_log/kibana_2022-08-19-01:00:12.log')
    # print('---'*10)
    # get_recall('../result_log/kibana_2022-08-19-01:06:14.log')
    # print('---'*10)
    # get_recall('../result_log/kibana_2022-08-19-01:13:26.log')
    # print('---'*10)
    # get_recall('../result_log/kibana_2022-08-19-01:20:52.log')

    ### vscode
    # get_recall('../result_log/vscode_2022-08-19-02:00:31.log')
    # print('---'*10)
    # get_recall('../result_log/vscode_2022-08-19-04:15:26.log')
    # print('---'*10)
    # get_recall('../result_log/vscode_2022-08-19-06:31:33.log')
    # print('---'*10)
    # get_recall('../result_log/vscode_2022-08-19-08:45:54.log')
    # print('---'*10)
    # get_recall('../result_log/vscode_2022-08-19-11:00:21.log')
    
    ### hadoop
    # get_recall('../result_log/hadoop_2022-08-18-22:44:57.log')
    # print('---'*10)
    # get_recall('../result_log/hadoop_2022-08-18-22:46:53.log')
    # print('---'*10)
    # get_recall('../result_log/hadoop_2022-08-19-13:18:14.log')
    # print('---'*10)
    # get_recall('../result_log/hadoop_2022-08-19-13:20:08.log')
    # print('---'*10)
    # get_recall('../result_log/hadoop_2022-08-19-13:22:03.log')
    
    ### spark
    # get_recall('../result_log/spark_2022-08-19-13:25:19.log')
    # print('---'*10)
    # get_recall('../result_log/spark_2022-08-19-13:26:46.log')
    # print('---'*10)
    # get_recall('../result_log/spark_2022-08-19-13:28:28.log')
    # print('---'*10)
    # get_recall('../result_log/spark_2022-08-19-13:29:53.log')
    # print('---'*10)
    # get_recall('../result_log/spark_2022-08-19-13:31:24.log')
    # print('eclipse')
    # get_recall('../result_log/.log')
    # print('...' * 10)
    
    # print('mozilla')
    # get_recall('../result_log/.log')
    # print('...' * 10)
    
    # print('eclipse')
    # get_recall('../result_log/eclipse_2022-08-25-14:04:58.log')
    # print('...' * 10)

    # print('mozilla')
    # get_recall('../result_log/mozilla_2022-08-24-22:37:03.log')
    # print('...' * 10)

    # print('hadoop')
    # get_recall('../result_log/hadoop_2022-08-25-00:09:54.log')
    # print('...' * 10)
    
    # print('spark')
    # get_recall('../result_log/spark_2022-08-25-00:12:01.log')
    # print('...' * 10)
    
    # print('kibana')
    # get_recall('../result_log/kibana_2022-08-25-09:23:06.log')
    # print('...' * 10)
    
    # print('vscode')
    # get_recall('../result_log/vscode_2022-08-24-22:03:09.log')
    # print('...' * 10)
    
    # get_recall('../result_log/eclipse-old_2022-08-29-22:31:54.log')

    print('eclipse')
    get_run_time('../result_log/random-RQ2/eclipse_2022-08-18-23:02:31.log')
    get_run_time('../result_log/random-RQ2/eclipse_2022-08-18-23:26:25.log')
    get_run_time('../result_log/random-RQ2/eclipse_2022-08-18-23:47:47.log')
    get_run_time('../result_log/random-RQ2/eclipse_2022-08-19-00:09:01.log')
    get_run_time('../result_log/random-RQ2/eclipse_2022-08-19-00:30:13.log')

    print('hadoop')
    get_run_time('../result_log/random-RQ2/hadoop_2022-08-18-22:44:57.log')
    get_run_time('../result_log/random-RQ2/hadoop_2022-08-18-22:46:53.log')
    get_run_time('../result_log/random-RQ2/hadoop_2022-08-19-13:18:14.log')
    get_run_time('../result_log/random-RQ2/hadoop_2022-08-19-13:20:08.log')
    get_run_time('../result_log/random-RQ2/hadoop_2022-08-19-13:22:03.log')

    print('kibana')
    get_run_time('../result_log/random-RQ2/kibana_2022-08-19-00:53:49.log')
    get_run_time('../result_log/random-RQ2/kibana_2022-08-19-01:00:12.log')
    get_run_time('../result_log/random-RQ2/kibana_2022-08-19-01:06:14.log')
    get_run_time('../result_log/random-RQ2/kibana_2022-08-19-01:13:26.log')
    get_run_time('../result_log/random-RQ2/kibana_2022-08-19-01:20:52.log')

    print('mozilla')
    get_run_time('../result_log/random-RQ2/mozilla_2022-08-20-09:45:36.log')
    get_run_time('../result_log/random-RQ2/mozilla_2022-08-21-15:35:19.log')
    get_run_time('../result_log/random-RQ2/mozilla_2022-08-22-16:16:37.log')
    get_run_time('../result_log/random-RQ2/mozilla_2022-08-22-21:35:15.log')
    get_run_time('../result_log/random-RQ2/mozilla_2022-08-24-22:37:03.log')

    print('spark')
    get_run_time('../result_log/random-RQ2/spark_2022-08-19-13:25:19.log')
    get_run_time('../result_log/random-RQ2/spark_2022-08-19-13:26:46.log')
    get_run_time('../result_log/random-RQ2/spark_2022-08-19-13:28:28.log')
    get_run_time('../result_log/random-RQ2/spark_2022-08-19-13:29:53.log')
    get_run_time('../result_log/random-RQ2/spark_2022-08-19-13:31:24.log')

    print('vscode')
    get_run_time('../result_log/random-RQ2/vscode_2022-08-19-02:00:31.log')
    get_run_time('../result_log/random-RQ2/vscode_2022-08-19-04:15:26.log')
    get_run_time('../result_log/random-RQ2/vscode_2022-08-19-06:31:33.log')
    get_run_time('../result_log/random-RQ2/vscode_2022-08-19-08:45:54.log')
    get_run_time('../result_log/random-RQ2/vscode_2022-08-19-11:00:21.log')