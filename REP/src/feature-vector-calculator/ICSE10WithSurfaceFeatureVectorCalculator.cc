/*
 * ICSE10WithSurfaceFeatureCalculator.cc
 *
 *  Created on: Dec 20, 2010
 *      Author: Chengnian SUN
 */

#include "ICSE10WithSurfaceFeatureVectorCalculator.h"
#include <algorithm>
#include <cmath>
#include "../feature-value-calculator/SurfaceFeatureValueCalculator.h"
using namespace std;

vector<AbstractFeatureValueCalculator*> ICSE10WithSurfaceFeatureVectorCalculator::create_Surface_feature_vector_calculators() const {
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

//, const unsigned textual_feature_count =
//ICSE10_FEATURE_COUNT, const unsigned surface_feature_count = 4

ICSE10WithSurfaceFeatureVectorCalculator::ICSE10WithSurfaceFeatureVectorCalculator(
    const ReportBuckets& report_buckets, const unsigned textual_feature_count,
    const unsigned surface_feature_count) :
    ICSE10FeatureVectorCalculator(report_buckets, textual_feature_count,
        surface_feature_count) {
}

ICSE10WithSurfaceFeatureVectorCalculator::ICSE10WithSurfaceFeatureVectorCalculator(
    const ReportBuckets& report_buckets) :
    ICSE10FeatureVectorCalculator(report_buckets, ICSE10_FEATURE_COUNT, 4) {

}

ICSE10WithSurfaceFeatureVectorCalculator::~ICSE10WithSurfaceFeatureVectorCalculator() {
}
