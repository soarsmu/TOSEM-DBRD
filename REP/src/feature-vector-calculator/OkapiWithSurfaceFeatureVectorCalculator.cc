/*
 * OkapiWithSurfaceFeatureCalculator.cc
 *
 *  Created on: Dec 21, 2010
 *      Author: Chengnian SUN
 */

#include "OkapiWithSurfaceFeatureVectorCalculator.h"
#include "../feature-value-calculator/SurfaceFeatureValueCalculator.h"

vector<AbstractFeatureValueCalculator*> OkapiWithSurfaceFeatureVectorCalculator::create_Surface_feature_vector_calculators() const {
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

OkapiWithSurfaceFeatureVectorCalculator::OkapiWithSurfaceFeatureVectorCalculator(
    const ReportBuckets& report_buckets) :
    OkapiFeatureVectorCalculator(report_buckets, ICSE10_FEATURE_COUNT, 4) {
}

OkapiWithSurfaceFeatureVectorCalculator::OkapiWithSurfaceFeatureVectorCalculator(
    const ReportBuckets& report_buckets, const unsigned textual_feature_count,
    const unsigned surface_feature_count) :
    OkapiFeatureVectorCalculator(report_buckets, textual_feature_count,
        surface_feature_count) {
}

OkapiWithSurfaceFeatureVectorCalculator::~OkapiWithSurfaceFeatureVectorCalculator() {
}
