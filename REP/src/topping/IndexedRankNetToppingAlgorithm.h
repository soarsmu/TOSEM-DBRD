/*
 * RankNetToppingAlgorithmWithIndex.h
 *
 *  Created on: Apr 4, 2013
 *      Author: neo
 */

#ifndef _RANKNET_TOPPING_ALGORITHM_WITH_INDEX_H_
#define _RANKNET_TOPPING_ALGORITHM_WITH_INDEX_H_

#include "RankNetToppingAlgorithm.h"

class InvertedIndex;
class Master2PartialBucket;

class IndexedRankNetToppingAlgorithm: public RankNetToppingAlgorithm {
private:

  unsigned m_size_of_indexed_reports;

  InvertedIndex* m_index;

  const float m_idf_threshold;

  void build_index();

  Master2PartialBucket* master_2_partial_bucket;

public:

  virtual void before_get_top();

  void internal_get_top(const AbstractBugReport& query_report,
      MasterReportPriorityQueue& queue);

  IndexedRankNetToppingAlgorithm(FILE* log_file, const ReportBuckets& buckets,
      const DefaultREPParameter& parameter);

  virtual ~IndexedRankNetToppingAlgorithm();
};

#endif /* _RANKNET_TOPPING_ALGORITHM_WITH_INDEX_H_ */
