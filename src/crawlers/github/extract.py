import logging
import os
import json

# Top 5 repos that have the largest number of bug report based on GHTorrent data as of March 2021
OWNER_REPO_DICT = {"nixos": "nixpkgs", "microsoft": "vscode", "elastic": "kibana",
                   "kubernetes": "kubernetes", "ansible": "ansible"}


def extract_edges(content: str):
    j = json.loads(content)
    return j['data']['repository']['issues']['edges']


def save_nodes(output_dir: str, edges: list):
    for edge in edges:
        node = edge['node']
        with open('%s/%s.json' % (output_dir, node['number']), "w") as output:
            output.write(json.dumps(node))


for owner, repo in OWNER_REPO_DICT.items():
    output_dir = './%s-%s-split/' % (owner, repo)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with open('./%s-%s/first.json' % (owner, repo)) as f:
        content = f.read()
        edges = extract_edges(content)
        save_nodes(output_dir, edges)
        last_cursor = edges[0]['cursor']

    while last_cursor:
        print(last_cursor)
        try:
            with open("./%s-%s/%s" % (owner, repo, last_cursor)) as f:
                content = f.read()
                edges = extract_edges(content)
                save_nodes(output_dir, edges)
                last_cursor = edges[0]['cursor']
        except Exception as e:
            logging.debug(e, exc_info=True)
            last_cursor = None
            continue
