/*
 * RanknetOkapiSurfaceFeatureVectorCalculator.cc
 *
 *  Created on: Jan 7, 2011
 *      Author: Chengnian SUN
 */

#include "RanknetOkapiSurfaceFeatureVectorCalculator.h"
#include "../feature-value-calculator/RankNetFeatureValueCalculator.h"

RanknetOkapiSurfaceFeatureVectorCalculator::RanknetOkapiSurfaceFeatureVectorCalculator(
    FILE* log_file, const ReportBuckets& report_buckets,
    const unsigned textual_feature_count, const unsigned surface_feature_count) :
    OkapiWithSurfaceFeatureVectorCalculator(report_buckets,
        textual_feature_count, surface_feature_count), m_log_file(log_file) {
}

vector<AbstractFeatureValueCalculator*> RanknetOkapiSurfaceFeatureVectorCalculator::create_Textual_feature_vector_calculators() const {
  vector<AbstractFeatureValueCalculator*> super =
      OkapiWithSurfaceFeatureVectorCalculator::create_Textual_feature_vector_calculators();
  super.push_back(
      new RankNetFeatureValueCalculator(this->m_log_file,
          this->get_report_buckets()));
  return super;
}

RanknetOkapiSurfaceFeatureVectorCalculator::RanknetOkapiSurfaceFeatureVectorCalculator(
    FILE* log_file, const ReportBuckets& report_buckets) :
    OkapiWithSurfaceFeatureVectorCalculator(report_buckets,
        ICSE10_FEATURE_COUNT + 1, 4), m_log_file(log_file) {

}

RanknetOkapiSurfaceFeatureVectorCalculator::~RanknetOkapiSurfaceFeatureVectorCalculator() {
}
