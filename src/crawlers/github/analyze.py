import json

last_cursor = None
with open('./microsoft-TypeScript/first.json') as f:
    content = f.read()
    j = json.loads(content)
    edges = j['data']['repository']['issues']['edges']
    for edge in reversed(edges):
        print(edge['node']['url'])
    last_cursor = edges[0]['cursor']

while last_cursor:
    print(last_cursor)
    with open("./microsoft-TypeScript/{}".format(last_cursor)) as f:
        content = f.read()
        j = json.loads(content)
        edges = j['data']['repository']['issues']['edges']
        for edge in reversed(edges):
                print(edge['node']['url'])
        last_cursor = edges[0]['cursor']

