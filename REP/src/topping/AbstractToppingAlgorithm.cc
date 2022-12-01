/*
 * AbstractToppingAlgorithm.cc
 *
 *  Created on: Jan 12, 2011
 *      Author: Chengnian SUN
 */

#include <boost/unordered_map.hpp>
#include <cstdio>

#include "AbstractToppingAlgorithm.h"

#include "../index/Master2PartialBucket.h"
#include "../report-model/DuplicateBugReport.h"
#include "../report-model/MasterBugReport.h"
#include "../report-model/SimilarityInfo.h"
#include "../detection-model/ReportBuckets.h"
#include "../detection-model/IndexingType.h"
#include "../util/CmdOption.h"
using boost::unordered_map;

AbstractToppingAlgorithm::~AbstractToppingAlgorithm() {

}

AbstractToppingAlgorithm::AbstractToppingAlgorithm(const ReportBuckets& buckets, FILE* log_file) :
    m_buckets(buckets), m_log_file(log_file), m_report_candidates_cache(
        CmdOption::get_instance().indexing_param_top_cosine_number()), m_time_constraint(
        CmdOption::get_instance().get_time_constraint()) {
}

std::pair<int, double> AbstractToppingAlgorithm:: compute_bucket_similarity(
    const AbstractBugReport& query_report, const MasterBugReport& master) {

  assert(master.get_id() != query_report.get_id());
  double max = this->compute_similarity(query_report, master);
  int report_id = master.get_id();

  double temp;
  const vector<const DuplicateBugReport*>& duplicates = master.get_duplicates();
  for (unsigned int i = 0, dup_size = duplicates.size(); i < dup_size; i++) {
    const DuplicateBugReport* dup = duplicates[i];

    assert(dup->get_id() != query_report.get_id());

    temp = this->compute_similarity(query_report, *dup);
    if (max < temp) {
      max = temp;
      report_id = dup->get_id();
    }
  }
  return std::make_pair(report_id, max);
}

void AbstractToppingAlgorithm::internal_get_top(const AbstractBugReport& query,
    MasterReportPriorityQueue& queue) {
  assert(this->m_report_candidates_cache.empty());
  const unsigned bucket_size = this->m_buckets.get_bucket_count();
  for (unsigned bucket_id = 0; bucket_id < bucket_size; bucket_id++) {
    MasterBugReport& master = *(this->m_buckets.get_bucket_master(bucket_id));
    const unsigned latest_timestamp = master.get_latest_timestamp_in_bucket();
    const unsigned delta = query.get_timestamp_in_days() - latest_timestamp;

    if (delta > this->m_time_constraint){
      if (master.get_id() == query.get_duplicate_id())
        fprintf(this->m_log_file, "INFO: the master set (%u) couldn't be reached (diff=%u,query=%u)\n", master.get_id() ,delta, query.get_id());
      continue;
    }
 
    master.get_similarity_info()->set_similarity(
        this->compute_bucket_similarity(query, master));
    queue.add(&master);
  }
}

void AbstractToppingAlgorithm::train_model(){
  this->before_get_top();
}

void AbstractToppingAlgorithm::get_top(const AbstractBugReport& query_report,
    vector<const MasterBugReport*>& result_collector) {
  assert(result_collector.empty());
  this->before_get_top();

  assert(this->m_queue_cache.empty());
//  this->m_buckets.reset_similarity();
  this->internal_get_top(query_report, this->m_queue_cache);

//  this->m_report_candidates_cache.clear();
//  assert(this->m_report_candidates_cache.empty());
//  const IndexingType::EnumIndexingType indexing_type =
//      this->m_buckets.get_report_candidates(query_report,
//          this->m_report_candidates_cache);
//  if (indexing_type == IndexingType::NO_INDEXING) {

//  } else {
//    assert(this->m_report_candidates_cache.size());
//
//    for (Master2PartialBucket::MapIterator iter =
//        this->m_report_candidates_cache.begin(), end =
//        this->m_report_candidates_cache.end(); iter != end; ++iter) {
//      const Master2PartialBucket::PartialBucket* bucket = iter->second;
//      assert(bucket->size());
//      Master2PartialBucket::PartialBucket::const_iterator biter =
//          bucket->begin();
//      const MasterBugReport* master = (*biter).get_report()->get_master();
//
//      double max = -9999;
//      int similar_report = -1;
//      for (Master2PartialBucket::PartialBucket::const_iterator bend =
//          bucket->end(); biter != bend; ++biter) {
//        const Master2PartialBucket::ReportWithCounter& element = *biter;
////        if (element.get_counter() < 2) {
////          continue;
////        }
//        const AbstractBugReport* report = element.get_report();
//        const int report_id = report->get_id();
//        if (this->m_report_candidates_cache.filtered(report_id))
//          continue;
//        const double report_sim = this->compute_similarity(query_report,
//            *report);
//        if (max < report_sim) {
//          max = report_sim;
//          similar_report = report_id;
//        }
//      }
//      master->get_similarity_info()->set_similarity(max, similar_report);
//      this->m_queue_cache.add(master);
//    }
//
//  }

  this->m_queue_cache.sort_and_desctroy(result_collector);
}
