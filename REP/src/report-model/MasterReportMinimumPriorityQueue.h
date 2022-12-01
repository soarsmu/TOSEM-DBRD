/*
 * MasterReportMinimumPriorityQueue.h
 *
 *  Created on: 2010-8-3
 *      Author: Chengnian Sun.
 */

#ifndef MASTERREPORTMINIMUMPRIORITYQUEUE_H_
#define MASTERREPORTMINIMUMPRIORITYQUEUE_H_

#include <cassert>
#include <queue>
#include <vector>

#include "../report-model/MasterBugReport.h"

using namespace std;

class MasterReportPriorityQueue {
private:
  struct MasterBugReportGreater {
    bool operator()(const MasterBugReport* r1,
        const MasterBugReport* r2) const {
      return r1->get_similarity_info()->get_similarity()
          > r2->get_similarity_info()->get_similarity();
    }
  };

  std::priority_queue<const MasterBugReport*, vector<const MasterBugReport*>,
      MasterBugReportGreater> m_queue;

public:

  inline void add(const MasterBugReport* master) {
    assert(master != NULL);
    this->m_queue.push(master);
  }

  inline bool empty() const {
    return this->m_queue.empty();
  }

  inline void sort_and_desctroy(
      vector<const MasterBugReport*>& result_collector) {
    assert(result_collector.empty());
    const unsigned size = this->m_queue.size();
    result_collector.assign(size, NULL);
    for (int i = size - 1; i > -1; i--) {
      result_collector[i] = this->m_queue.top();
      this->m_queue.pop();
    }
    assert(this->m_queue.empty());
  }

  MasterReportPriorityQueue() {
  }

  ~MasterReportPriorityQueue() {

  }
};

#endif /* MASTERREPORTMINIMUMPRIORITYQUEUE_H_ */
