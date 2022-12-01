import os
import json
import requests
import logging

# Import and init common logger
import sys
sys.path.append("../")
from logger import init_logger
init_logger()

# The authz token is a temporary personal access token for testing

# Top 5 repos that have the largest number of bug report based on GHTorrent data as of March 2021
OWNER_REPO_DICT = {"microsoft": "vscode", 
                   "elastic": "kibana"}

query_template = """
{
  repository(owner: "%s", name: "%s") {
    issues(last: %s %s) {
      edges {
        cursor
        node {
          author {
            login
          }
          bodyText
          createdAt
          closed
          closedAt
          comments(first: 100) {
            totalCount
            nodes {
              author {
                login
              }
              body
              createdAt
              lastEditedAt
              publishedAt
              updatedAt
              url
            }
          }
          isPinned
          url
          title
          updatedAt
          publishedAt
          lastEditedAt
          number
          state
          labels(first: 100) {
            nodes {
              createdAt
              name
            }
            totalCount
          }
          timelineItems(first: 100) {
            totalCount
            nodes {
              __typename
              ... on LabeledEvent {
                actor {
                  login
                }
                createdAt
                label {
                  name
                }
              }
              ... on MarkedAsDuplicateEvent {
                actor {
                  login
                }
                createdAt
                canonical {
                  ... on Issue {
                    number
                    url
                    createdAt
                  }
                  ... on PullRequest {
                    number
                    createdAt
                  }
                }
              }
              ... on UnlabeledEvent {
                actor {
                  login
                }
                createdAt
                label {
                  name
                }
              }
              ... on UnmarkedAsDuplicateEvent {
                id
                actor {
                  login
                }
                createdAt
                canonical {
                  ... on Issue {
                    number
                    url
                    createdAt
                  }
                  ... on PullRequest {
                    number
                    createdAt
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  rateLimit {
    limit
    cost
    remaining
    resetAt
  }
}
"""


def run_query(q: str) -> dict:
    request = requests.post('https://api.github.com/graphql', json={'query': q}, headers=HEADERS)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, q))


def generate_query(repo_owner: str, name: str, issues_before_cursor: str = None, num_issues: int = 50) -> str:
    before_cursor = ', before: "%s"' % issues_before_cursor if issues_before_cursor else ""
    try:
        q = query_template % (repo_owner, name, str(num_issues), before_cursor)
    except Exception as e:
        logging.error("Query failed", e, exc_info=True)
        # return with smaller number of issues
        return generate_query(repo_owner, name, issues_before_cursor, num_issues - 10)
    return q


def crawl(repo_owner: str, name: str, output_directory: str,
          issues_before_cursor: str = None, num_issues: int = 50) -> str:
    query = generate_query(repo_owner, name, issues_before_cursor, num_issues)
    try:
        result = run_query(query)
    except Exception as e:
        logging.error("Query failed", e, exc_info=True)
        return crawl(repo_owner, name, output_directory, issues_before_cursor, num_issues-20)
    new_cursor = None

    output_name = output_directory + issues_before_cursor if issues_before_cursor else output_dir + "first.json"
    with open(output_name, "w") as f:
        f.write(json.dumps(result))

    edges = result['data']['repository']['issues']['edges']
    if len(edges) == num_issues:
        new_cursor = edges[0]['cursor']
        logging.info("Next cursor: %s" % new_cursor)

    remaining_rate_limit = result["data"]["rateLimit"]["remaining"]  # Drill down the dictionary
    logging.info("Remaining rate limit - {}".format(remaining_rate_limit))

    return new_cursor


for owner, repo in OWNER_REPO_DICT.items():
    output_dir = './%s-%s/' % (owner, repo)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    cursor = crawl(owner, repo, output_dir, None)
    while cursor:
        cursor = crawl(owner, repo, output_dir, cursor)
    logging.info("Crawling issues from '%s/%s' is done" % (owner, repo))
