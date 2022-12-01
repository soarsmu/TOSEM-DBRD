import re

def get_recall(file):
    with open(file, 'r') as f:
        data = f.read()

    tmp_list = re.findall(r"\'rate\': 0\.\d+", data)[:10]
    
    for i, content in zip(range(len(tmp_list)), tmp_list):
        print('{}'.format(re.findall(r"0\.\d+", content)[0]))
    print('====='*10)

def get_run_time(file):
    with open(file, 'r') as f:
        data = f.read()

    tmp_list = re.findall(r"seconds=\d+\.\d+", data)[0]
    print('{}'.format(re.findall(r"\d+\.\d+", tmp_list)[0]))


# get_run_time('result_log/discussion-state-jira/sabd_spark_2022-08-16-11:07:47.log')
# get_run_time('result_log/discussion-state-jira/sabd_spark_2022-08-16-11:09:41.log')
# get_run_time('result_log/discussion-state-jira/sabd_spark_2022-08-16-11:11:02.log')
# get_run_time('result_log/discussion-state-jira/sabd_spark_2022-08-16-11:12:25.log')
# get_run_time('result_log/discussion-state-jira/sabd_spark_2022-08-16-13:10:13.log')

# get_run_time('result_log/discussion-state-jira/sabd_hadoop_2022-08-16-11:13:43.log')
# get_run_time('result_log/discussion-state-jira/sabd_hadoop_2022-08-16-11:15:14.log')
# get_run_time('result_log/discussion-state-jira/sabd_hadoop_2022-08-16-11:16:46.log')
# get_run_time('result_log/discussion-state-jira/sabd_hadoop_2022-08-16-11:18:18.log')
# get_run_time('result_log/discussion-state-jira/sabd_hadoop_2022-08-16-11:19:55.log')

# get_run_time('result_log/RQ1-ITS/sabd_kibana_2022-08-17-19:56:51.log')
# get_run_time('result_log/RQ1-ITS/sabd_kibana_2022-08-17-20:01:31.log')
# get_run_time('result_log/RQ1-ITS/sabd_kibana_2022-08-17-20:06:11.log')
# get_run_time('result_log/RQ1-ITS/sabd_kibana_2022-08-17-20:10:57.log')
# get_run_time('result_log/RQ1-ITS/sabd_kibana_2022-08-17-20:15:43.log')

# get_run_time('result_log/RQ1-ITS/sabd_vscode_2022-08-17-19:57:28.log')
# get_run_time('result_log/RQ1-ITS/sabd_vscode_2022-08-17-21:32:47.log')
# get_run_time('result_log/RQ1-ITS/sabd_vscode_2022-08-17-23:11:18.log')
# get_run_time('result_log/RQ1-ITS/sabd_vscode_2022-08-18-00:46:15.log')
# get_run_time('result_log/RQ1-ITS/sabd_vscode_2022-08-18-02:15:35.log')

# get_run_time('result_log/RQ1-Age-Bugzilla/sabd_eclipse_2022-08-17-14:39:15.log')
# get_run_time('result_log/RQ1-Age-Bugzilla/sabd_eclipse_2022-08-17-14:54:12.log')
# get_run_time('result_log/RQ1-Age-Bugzilla/sabd_eclipse_2022-08-17-15:09:20.log')
# get_run_time('result_log/RQ1-Age-Bugzilla/sabd_eclipse_2022-08-17-15:24:31.log')
# get_run_time('result_log/RQ1-Age-Bugzilla/sabd_eclipse_2022-08-17-15:40:04.log')

get_run_time('result_log/RQ1-Age-Bugzilla/sabd_mozilla_2022-08-17-14:46:57.log')
get_run_time('result_log/RQ1-Age-Bugzilla/sabd_mozilla_2022-08-18-14:00:20.log')
get_run_time('result_log/RQ1-Age-Bugzilla/sabd_mozilla_2022-08-19-13:02:55.log')
get_run_time('result_log/RQ1-Age-Bugzilla/sabd_mozilla_2022-08-20-12:08:48.log')
get_run_time('result_log/RQ1-Age-Bugzilla/sabd_mozilla_2022-08-21-11:05:15.log')

## Age
# get_recall('result_log/discussion-sample/sabd_eclipse-sampled_2022-08-30-07:03:56.log')
# get_recall('result_log/discussion-sample/sabd_eclipse-sampled_2022-08-31-20:15:58.log')
# get_recall('result_log/discussion-sample/sabd_eclipse-sampled_2022-08-31-20:17:02.log')
# get_recall('result_log/discussion-sample/sabd_eclipse-sampled_2022-08-31-20:17:39.log')
# get_recall('result_log/discussion-sample/sabd_eclipse-sampled_2022-08-31-20:34:12.log')

# get_recall('result_log/discussion-sample/sabd_eclipse-old-sampled-age_2022-08-26-23:31:25.log')
# get_recall('result_log/discussion-sample/sabd_eclipse-old-sampled-age_2022-08-26-23:34:35.log')
# get_recall('result_log/discussion-sample/sabd_eclipse-old-sampled-age_2022-08-26-23:38:23.log')
# get_recall('result_log/discussion-sample/sabd_eclipse-old-sampled-age_2022-08-27-04:54:07.log')
# get_recall('result_log/discussion-sample/sabd_eclipse-old-sampled-age_2022-08-31-21:13:40.log')

# get_recall('result_log/discussion-sample/pairs_eclipse-sampled_2022-08-30-14:19:12.log')
# get_recall('result_log/discussion-sample/pairs_eclipse-sampled_2022-08-30-14:19:24.log')
# get_recall('result_log/discussion-sample/pairs_eclipse-sampled_2022-08-30-14:29:30.log')
# get_recall('result_log/discussion-sample/pairs_eclipse-sampled_2022-08-30-14:29:34.log')
# get_recall('result_log/discussion-sample/pairs_eclipse-sampled_2022-08-30-14:39:53.log')


# get_recall('result_log/discussion-sample/pairs_eclipse-old-sampled_2022-08-26-21:37:08.log')
# get_recall('result_log/discussion-sample/pairs_eclipse-old-sampled_2022-08-26-22:24:51.log')
# get_recall('result_log/discussion-sample/pairs_eclipse-old-sampled_2022-08-26-23:33:04.log')
# get_recall('result_log/discussion-sample/pairs_eclipse-old-sampled_2022-08-26-23:37:01.log')
# get_recall('result_log/discussion-sample/pairs_eclipse-old-sampled_2022-08-26-23:38:50.log')



# get_recall('result_log/discussion-sample/pairs_mozilla-sampled_2022-08-27-09:45:56.log')
# get_recall('result_log/discussion-sample/pairs_mozilla-sampled_2022-08-27-10:51:11.log')
# get_recall('result_log/discussion-sample/pairs_mozilla-sampled_2022-08-27-12:30:14.log')
# get_recall('result_log/discussion-sample/pairs_mozilla-sampled_2022-08-30-08:29:51.log')
# get_recall('result_log/discussion-sample/pairs_mozilla-sampled_2022-08-30-08:30:10.log')


# get_recall('result_log/discussion-sample/pairs_mozilla-old-sampled_2022-08-27-04:05:37.log')
# get_recall('result_log/discussion-sample/pairs_mozilla-old-sampled_2022-08-27-04:31:03.log')
# get_recall('result_log/discussion-sample/pairs_mozilla-old-sampled_2022-08-27-04:39:17.log')
# get_recall('result_log/discussion-sample/pairs_mozilla-old-sampled_2022-08-30-00:39:15.log')
# get_recall('result_log/discussion-sample/pairs_mozilla-old-sampled_2022-08-30-10:32:52.log')


# get_recall('result_log/discussion-sample/sabd_mozilla-sampled_2022-08-27-15:58:57.log')
# get_recall('result_log/discussion-sample/sabd_mozilla-sampled_2022-08-28-12:25:27.log')
# get_recall('result_log/discussion-sample/sabd_mozilla-sampled_2022-08-28-22:03:22.log')
# get_recall('result_log/discussion-sample/sabd_mozilla-sampled_2022-08-28-22:03:43.log')
# get_recall('result_log/discussion-sample/sabd_mozilla-sampled_2022-08-29-00:11:27.log')


# get_recall('result_log/discussion-sample/sabd_mozilla-old-sampled_2022-08-26-18:18:47.log')
# get_recall('result_log/discussion-sample/sabd_mozilla-old-sampled_2022-08-27-04:45:47.log')
# get_recall('result_log/discussion-sample/sabd_mozilla-old-sampled_2022-08-27-15:56:00.log')
# get_recall('result_log/discussion-sample/sabd_mozilla-old-sampled_2022-08-27-15:56:48.log')
# get_recall('result_log/discussion-sample/sabd_mozilla-old-sampled_2022-08-27-15:58:39.log')


# ITS - Jira
# get_recall('result_log/discussion-sample/pairs_eclipse-sampled-jira_2022-08-26-18:50:19.log')
# get_recall('result_log/discussion-sample/pairs_eclipse-sampled-jira_2022-08-26-20:31:30.log')
# get_recall('result_log/discussion-sample/pairs_eclipse-sampled-jira_2022-08-27-02:00:51.log')
# get_recall('result_log/discussion-sample/pairs_eclipse-sampled-jira_2022-08-27-02:10:11.log')
# get_recall('result_log/discussion-sample/pairs_eclipse-sampled-jira_2022-08-27-02:33:08.log')


# get_recall('result_log/discussion-sample/pairs_mozilla-sampled-jira_2022-08-27-02:18:44.log')
# get_recall('result_log/discussion-sample/pairs_mozilla-sampled-jira_2022-08-27-02:31:22.log')
# get_recall('result_log/discussion-sample/pairs_mozilla-sampled-jira_2022-08-27-02:54:36.log')
# get_recall('result_log/discussion-sample/pairs_mozilla-sampled-jira_2022-08-30-10:49:53.log')
# get_recall('result_log/discussion-sample/pairs_mozilla-sampled-jira_2022-08-30-10:50:05.log')


# get_recall('result_log/pairs_hadoop-sampled_2022-08-27-02:10:12.log')
# get_recall('result_log/pairs_hadoop-sampled_2022-08-27-02:21:39.log')
# get_recall('result_log/pairs_hadoop-sampled_2022-08-27-02:43:26.log')


# get_recall('result_log/pairs_spark-sampled_2022-08-27-03:57:05.log')
# get_recall('result_log/pairs_spark-sampled_2022-08-27-04:20:34.log')
# get_recall('result_log/pairs_spark-sampled_2022-08-27-04:30:14.log')



# get_recall('result_log/discussion-sample/sabd_eclipse-sampled-jira_2022-08-26-22:33:10.log')
# get_recall('result_log/discussion-sample/sabd_eclipse-sampled-jira_2022-08-28-04:08:07.log')
# get_recall('result_log/discussion-sample/sabd_eclipse-sampled-jira_2022-08-28-06:01:18.log')
# get_recall('result_log/discussion-sample/sabd_eclipse-sampled-jira_2022-08-28-06:02:38.log')
# get_recall('result_log/discussion-sample/sabd_eclipse-sampled-jira_2022-08-31-13:10:10.log')


# get_recall('result_log/discussion-sample/sabd_mozilla-sampled-jira_2022-08-27-11:38:00.log')
# get_recall('result_log/discussion-sample/sabd_mozilla-sampled-jira_2022-08-27-22:46:33.log')
# get_recall('result_log/discussion-sample/sabd_mozilla-sampled-jira_2022-08-27-22:47:48.log')
# get_recall('result_log/discussion-sample/sabd_mozilla-sampled-jira_2022-08-28-04:23:31.log')
# get_recall('result_log/discussion-sample/sabd_mozilla-sampled-jira_2022-08-28-06:17:55.log')



# get_recall('result_log/discussion-sample/sabd_hadoop-sampled_2022-08-26-19:58:25.log')
# get_recall('result_log/discussion-sample/sabd_hadoop-sampled_2022-08-26-22:55:12.log')
# get_recall('result_log/discussion-sample/sabd_hadoop-sampled_2022-08-28-04:22:01.log')
# get_recall('result_log/discussion-sample/sabd_hadoop-sampled_2022-08-28-06:16:19.log')
# get_recall('result_log/discussion-sample/sabd_hadoop-sampled_2022-08-28-06:17:38.log')


# get_recall('result_log/discussion-sample/sabd_spark-sampled_2022-08-26-20:00:33.log')
# get_recall('result_log/discussion-sample/sabd_spark-sampled_2022-08-26-22:57:36.log')
# get_recall('result_log/discussion-sample/sabd_spark-sampled_2022-08-29-02:04:01.log')
# get_recall('result_log/discussion-sample/sabd_spark-sampled_2022-08-29-05:15:49.log')
# get_recall('result_log/discussion-sample/sabd_spark-sampled_2022-08-30-14:26:10.log')





## ITS - GitHub
# get_recall('result_log/discussion-sample/pairs_eclipse-sampled-github_2022-08-26-18:03:22.log')
# get_recall('result_log/discussion-sample/pairs_eclipse-sampled-github_2022-08-27-00:02:47.log')
# get_recall('result_log/discussion-sample/pairs_eclipse-sampled-github_2022-08-27-00:03:43.log')
# get_recall('result_log/discussion-sample/pairs_eclipse-sampled-github_2022-08-27-00:06:23.log')
# get_recall('result_log/discussion-sample/pairs_eclipse-sampled-github_2022-08-27-16:05:09.log')


# get_recall('result_log/discussion-sample/sabd_eclipse-sampled-github_2022-08-26-17:52:12.log')
# get_recall('result_log/discussion-sample/sabd_eclipse-sampled-github_2022-08-26-20:02:22.log')
# get_recall('result_log/discussion-sample/sabd_eclipse-sampled-github_2022-08-27-04:13:23.log')
# get_recall('result_log/discussion-sample/sabd_eclipse-sampled-github_2022-08-27-04:33:56.log')
# get_recall('result_log/discussion-sample/sabd_eclipse-sampled-github_2022-08-27-04:34:40.log')


# get_recall('result_log/discussion-sample/pairs_mozilla-sampled-github_2022-08-27-00:16:40.log')
# get_recall('result_log/discussion-sample/pairs_mozilla-sampled-github_2022-08-27-16:59:44.log')
# get_recall('result_log/discussion-sample/pairs_mozilla-sampled-github_2022-08-27-16:59:51.log')
# get_recall('result_log/discussion-sample/pairs_mozilla-sampled-github_2022-08-27-18:51:14.log')
# get_recall('result_log/discussion-sample/pairs_mozilla-sampled-github_2022-08-27-19:01:54.log')


# get_recall('result_log/discussion-sample/sabd_mozilla-sampled-github_2022-08-27-04:27:28.log')
# get_recall('result_log/discussion-sample/sabd_mozilla-sampled-github_2022-08-27-04:48:38.log')
# get_recall('result_log/discussion-sample/sabd_mozilla-sampled-github_2022-08-27-04:49:24.log')
# get_recall('result_log/discussion-sample/sabd_mozilla-sampled-github_2022-08-29-06:38:32.log')
# get_recall('result_log/discussion-sample/sabd_mozilla-sampled-github_2022-08-29-14:33:53.log')


# get_recall('result_log/discussion-sample/pairs_kibana-sampled_2022-08-30-19:46:31.log')
# get_recall('result_log/discussion-sample/pairs_kibana-sampled_2022-08-30-19:57:42.log')
# get_recall('result_log/discussion-sample/pairs_kibana-sampled_2022-08-30-20:09:10.log')
# get_recall('result_log/discussion-sample/pairs_kibana-sampled_2022-08-30-20:18:20.log')
# get_recall('result_log/discussion-sample/pairs_kibana-sampled_2022-08-30-20:26:52.log')


# get_recall('result_log/discussion-sample/sabd_kibana-sampled_2022-08-30-20:35:21.log')
# get_recall('result_log/discussion-sample/sabd_kibana-sampled_2022-08-30-20:40:00.log')
# get_recall('result_log/discussion-sample/sabd_kibana-sampled_2022-08-30-20:44:48.log')
# get_recall('result_log/discussion-sample/sabd_kibana-sampled_2022-08-30-20:49:29.log')
# get_recall('result_log/discussion-sample/sabd_kibana-sampled_2022-08-30-20:54:09.log')


# get_recall('result_log/pairs_vscode-sampled_2022-08-27-01:46:13.log')
# get_recall('result_log/pairs_vscode-sampled_2022-08-27-01:51:59.log')
# get_recall('result_log/pairs_vscode-sampled_2022-08-27-02:17:48.log')


# get_recall('result_log/discussion-sample/sabd_vscode-sampled_2022-08-26-18:09:29.log')
# get_recall('result_log/discussion-sample/sabd_vscode-sampled_2022-08-26-20:22:48.log')
# get_recall('result_log/discussion-sample/sabd_vscode-sampled_2022-08-28-02:48:07.log')
# get_recall('result_log/discussion-sample/sabd_vscode-sampled_2022-08-28-04:34:53.log')
# get_recall('result_log/discussion-sample/sabd_vscode-sampled_2022-08-28-04:36:05.log')