/*
 * RankNetFeatureValueCalculator.cc
 *
 *  Created on: Jan 7, 2011
 *      Author: Chengnian SUN
 */

#include "RankNetFeatureValueCalculator.h"
#include "../ranknet/BM25FRankNetLearner.h"
#include <cassert>
double RankNetFeatureValueCalculator::compute_feature_value(
    const AbstractBugReport& query_report, const AbstractBugReport& report) {
  if (this->m_leaner == NULL) {
    //XXX FIXME, TODO, this is a bug here.
    assert(false);
//		this->m_leaner = new BM25FRankNetLearner(this->m_log_file, this->get_report_buckets());
    this->m_leaner->learn();
  }
  return this->m_leaner->compute_similarity(query_report, report);
}

RankNetFeatureValueCalculator::RankNetFeatureValueCalculator(FILE* log_file,
    const ReportBuckets& buckets) :
    AbstractFeatureValueCalculator(buckets), m_leaner(NULL), m_log_file(
        log_file) {
}

RankNetFeatureValueCalculator::~RankNetFeatureValueCalculator() {
}
