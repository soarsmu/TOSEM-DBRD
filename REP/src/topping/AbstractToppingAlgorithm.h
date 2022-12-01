/*
 * AbstractToppingAlgorithm.h
 *
 *  Created on: 2010-8-3
 *      Author: Chengnian Sun.
 */

#ifndef ABSTRACTTOPPINGALGORITHM_H_
#define ABSTRACTTOPPINGALGORITHM_H_

#include <boost/unordered_map.hpp>
#include <utility>
#include <vector>

#include "../index/Master2PartialBucket.h"
#include "../report-model/MasterReportMinimumPriorityQueue.h"

using namespace std;

//class Master2PartialBucket;
class MasterBugReport;
class DuplicateBugReport;
class ReportBuckets;

class AbstractToppingAlgorithm {
private:

  const ReportBuckets& m_buckets;

  MasterReportPriorityQueue m_queue_cache;

  Master2PartialBucket m_report_candidates_cache;

  std::pair<int, double> compute_bucket_similarity(
      const AbstractBugReport& query_report, const MasterBugReport& master);

  const unsigned m_time_constraint;

protected:
  FILE* m_log_file;

  const ReportBuckets& get_buckets() const;

  virtual void before_get_top() = 0;

  virtual double compute_similarity(const AbstractBugReport& query_report,
      const AbstractBugReport& report) = 0;

  virtual void internal_get_top(const AbstractBugReport& query_report,
      MasterReportPriorityQueue& queue);

public:

  /**
   * the result collector should be empty.
   */
  void get_top(const AbstractBugReport& query_report,
      vector<const MasterBugReport*>& result_collector);

  void train_model();

  AbstractToppingAlgorithm(const ReportBuckets& buckets, FILE* log_file);

  virtual ~AbstractToppingAlgorithm();
};

inline const ReportBuckets& AbstractToppingAlgorithm::get_buckets() const {
  return this->m_buckets;
}

#endif /* ABSTRACTTOPPINGALGORITHM_H_ */
