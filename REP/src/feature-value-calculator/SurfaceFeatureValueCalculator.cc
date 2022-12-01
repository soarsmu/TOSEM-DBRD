/*
 * SurfaceFeatureValueCalculator.cc
 *
 *  Created on: Jan 7, 2011
 *      Author: Chengnian SUN
 */

#include "SurfaceFeatureValueCalculator.h"
#include "../util/MacroUtility.h"

double SurfaceFeatureValueCalculator::compute_feature_value(
    const AbstractBugReport& query_report,
    const AbstractBugReport& base_report) {
  int feature_value = 0;

  switch (this->m_type) {
  case VERSION:
    //		UNREACHABLE("unhandled surface feature [version]")
    feature_value = compute_version_similarity(query_report, base_report);
    break;
  case COMPONENT:
    feature_value = compute_component_similarity(query_report, base_report);
    break;
  case SUB_COMPONENT:
    feature_value = compute_sub_component_similarity(query_report, base_report);
    break;
  case REPORT_TYPE:
    feature_value = compute_report_type_similarity(query_report, base_report);
    break;
  case PRIORITY:
    //			feature_value = query_report.get_priority() - base_report.get_priority();
    //			if (feature_value < 0) {
    //				feature_value = -feature_value;
    //			}
    feature_value = compute_priority_similarity(query_report, base_report);
    break;
  default:
    UNREACHABLE("unhandled surface feature type.");
    break;
  }

  return feature_value;
}

SurfaceFeatureValueCalculator::SurfaceFeatureValueCalculator(
    const ReportBuckets& report_buckets, EnumSurfaceFeatureType type) :
    AbstractFeatureValueCalculator(report_buckets), m_type(type) {
}

SurfaceFeatureValueCalculator::~SurfaceFeatureValueCalculator() {
}
