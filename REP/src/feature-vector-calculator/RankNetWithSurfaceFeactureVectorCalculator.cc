/*
 * RankNetWithSurfaceFeactureVectorCalculator.cc
 *
 *  Created on: Jan 7, 2011
 *      Author: Chengnian SUN
 */

#include "RankNetWithSurfaceFeactureVectorCalculator.h"
#include "../feature-value-calculator/RankNetFeatureValueCalculator.h"
#include "../feature-value-calculator/SurfaceFeatureValueCalculator.h"

vector<AbstractFeatureValueCalculator*> RankNetWithSurfaceFeactureVectorCalculator::create_Textual_feature_vector_calculators() const {
  vector<AbstractFeatureValueCalculator*> result;

  result.push_back(
      new RankNetFeatureValueCalculator(this->m_log_file,
          this->get_report_buckets()));

  return result;
}

vector<AbstractFeatureValueCalculator*> RankNetWithSurfaceFeactureVectorCalculator::create_Surface_feature_vector_calculators() const {
  vector<AbstractFeatureValueCalculator*> result;
  const ReportBuckets& report_buckets = this->get_report_buckets();
  result.push_back(
      new SurfaceFeatureValueCalculator(report_buckets,
          SurfaceFeatureValueCalculator::COMPONENT));
  result.push_back(
      new SurfaceFeatureValueCalculator(report_buckets,
          SurfaceFeatureValueCalculator::SUB_COMPONENT));
  result.push_back(
      new SurfaceFeatureValueCalculator(report_buckets,
          SurfaceFeatureValueCalculator::REPORT_TYPE));
  result.push_back(
      new SurfaceFeatureValueCalculator(report_buckets,
          SurfaceFeatureValueCalculator::PRIORITY));
  return result;
}

RankNetWithSurfaceFeactureVectorCalculator::RankNetWithSurfaceFeactureVectorCalculator(
    FILE* log_file, const ReportBuckets& report_buckets,
    const unsigned textual_feature_count, const unsigned surface_feature_count) :
    AbstractFeatureVectorCalculator(report_buckets, textual_feature_count,
        surface_feature_count), m_log_file(log_file) {
}

RankNetWithSurfaceFeactureVectorCalculator::RankNetWithSurfaceFeactureVectorCalculator(
    FILE* log_file, const ReportBuckets& report_buckets) :
    AbstractFeatureVectorCalculator(report_buckets, 1, 4), m_log_file(log_file) {
}

RankNetWithSurfaceFeactureVectorCalculator::~RankNetWithSurfaceFeactureVectorCalculator() {
}
