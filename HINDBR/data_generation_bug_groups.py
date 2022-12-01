import logging
import networkx as nx
import pickle
import json
import argparse
import os

"""
Generate bug groups
identify non-duplicate bugs as master bug reports, then storing them in dictionary bug_group
- each item represents the same bug. 
- key: master bug report
- value: duplicate bug report(s) or none if the master bug report has no duplicates
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter project name.')
    parser.add_argument('--project', help='project name', required=True)

    args = parser.parse_args()
    PROJECT = args.project

    threshold_year = '2020'
    if 'old' in PROJECT:
        threshold_year = '2014'
        
    json_file = '../SABD/dataset/{}/{}.json'.format(PROJECT, PROJECT)
    with open(json_file) as f:
        lines = f.readlines()
        
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('Generating bug groups for the project: ' + PROJECT)

    os.makedirs('data/bug_report_groups/', exist_ok=True)
    BUG_GROUP_FNAME = 'data/bug_report_groups/' + PROJECT + '_all.pkl'

    bug_graph = nx.DiGraph()

    xmlfile_count = 0
    bug_id_content = dict()

    for line in lines:
        cur_bug = json.loads(line)
        if cur_bug['creation_ts'] >= threshold_year:
            print('the current bug is out of training scope {}'.format(cur_bug['bug_id']))
            break
        bug_id_content[cur_bug['bug_id']] = cur_bug
        
        xmlfile_count += 1
        if xmlfile_count % 1000 == 0:
            logging.info('Processing {0} xml files'.format(xmlfile_count))
        
        # get bug id
        bugid = int(cur_bug['bug_id'])              
        
        # get resolution status
        # resolution = cur_bug['resolution']
        # get dupids if the resolution is duplicate
        # if resolution == 'DUPLICATE':
        if len(cur_bug['dup_id']) > 0:
            dupid = cur_bug['dup_id'] 
            # dupid found
            if len(dupid) != 0:
                # the reported time of the dupid may beyond the selected data set
                # add edge (bug_id (resolution: DUPLICATE) -> bug_id (dup_id: master candidate)) to bug_graph
                dupid = int(dupid)
                if dupid == bugid:
                    bug_graph.add_node(bugid)
                    continue
                bug_graph.add_edge(bugid, dupid)  

            # The dupids are missing in some bug reports, thus we have to discard these bug reports from the data set.
            else:
                pass
        else:
            bug_graph.add_node(bugid)

    logging.info('Note that {0} bug report xmlfiles in the dataset'.format(xmlfile_count))
    logging.info('Note that {0} valid bug reports in the bug group {1}'.format(len(bug_graph), BUG_GROUP_FNAME))

    bug_group = dict()

    for subgraph in nx.weakly_connected_component_subgraphs(bug_graph):
        # one bug report in the subgraph: non-duplicate
        if len(subgraph) == 1:  
            non_duplicate_bug_report = list(subgraph.nodes()).pop()
            bug_group[non_duplicate_bug_report] = set()
            # more than one bug report in the subgraph: dupliccates, including two cases: cycle or no cycle
        else:   
            master_candidates = list(subgraph.nodes())
            # check whether the duplicate subgraph has cycles
            try:  
                cycle_edges = nx.algorithms.cycles.find_cycle(subgraph)
            # subgraph has no cycles
            except:  
                for out_degree in bug_graph.out_degree(node for node in master_candidates):
                    if out_degree[1] == 0:
                        # master bug report has no dupid (out edge), thus its out degree is 0
                        master_bug_report = out_degree[0]  
                # all duplicates without master bug report
                master_candidates.remove(master_bug_report)  
                bug_group[master_bug_report] = set()

                for duplicate in master_candidates:
                    bug_group[master_bug_report].add(duplicate)

            # subgraph has a cycle (e.g., bugid (dupid) -> dupid (bugid)
            else: 
                # cycle issue has already been handled by SABD
                pass
            

    f = open(BUG_GROUP_FNAME,'wb')
    pickle.dump(bug_group,f)
    f.close()

    logging.info('Note that {0} master bug reports in the bug group {1}'.format(len(bug_group), BUG_GROUP_FNAME))
    logging.info('Note that {0} duplicates in the bug group {1}'.format(sum(len(bug_group[key]) for key in bug_group), BUG_GROUP_FNAME))