/*
 * SurfaceFeatureValueCalculator.h
 *
 *  Created on: Jan 7, 2011
 *      Author: Chengnian SUN
 */

#ifndef SURFACEFEATUREVALUECALCULATOR_H_
#define SURFACEFEATUREVALUECALCULATOR_H_

#include "AbstractFeatureValueCalculator.h"
#include <cmath>

#include "../report-model/MasterBugReport.h"
#include "../report-model/DuplicateBugReport.h"

class SurfaceFeatureValueCalculator: public AbstractFeatureValueCalculator {
public:

  enum EnumSurfaceFeatureType {

    VERSION,

    COMPONENT,

    SUB_COMPONENT,

    REPORT_TYPE,

    PRIORITY
  };

private:
  EnumSurfaceFeatureType m_type;

public:

  static float compute_version_similarity(
      const AbstractBugReport& query_report,
      const AbstractBugReport& base_report) {
    int distance = query_report.get_version() - base_report.get_version();
    if (distance < 0) {
      distance = -distance;
    }
    assert(distance == distance);
    assert(distance >= 0);
    //		const double result = 1.0 / 1 + log(1.0 + distance);
    const float result = 1.0f / (1.0f + distance);
    assert(result == result);
    return result;
  }

  static float compute_component_similarity(
      const AbstractBugReport& query_report,
      const AbstractBugReport& base_report) {
    return (query_report.get_component() == base_report.get_component());
  }

  static float compute_sub_component_similarity(
      const AbstractBugReport& query_report,
      const AbstractBugReport& base_report) {
    return (query_report.get_sub_component() == base_report.get_sub_component());
  }

  static float compute_report_type_similarity(
      const AbstractBugReport& query_report,
      const AbstractBugReport& base_report) {
    return (query_report.get_report_type() == base_report.get_report_type());
  }

  static float compute_priority_similarity(
      const AbstractBugReport& query_report,
      const AbstractBugReport& base_report) {
//		double feature_value = query_report.get_priority() - base_report.get_priority();
//		if (feature_value < 0) {
//			feature_value = -feature_value;
//		}
//		return feature_value;

    int distance = query_report.get_priority() - base_report.get_priority();
    if (distance < 0) {
      distance = -distance;
    }
    assert(distance == distance);
    assert(distance >= 0);
    //		const double result = 1.0 / 1 + log(1.0 + distance);
    const float result = 1.0f / (1.0f + distance);
    assert(result == result);
    return result;
  }

  virtual double compute_feature_value(const AbstractBugReport& query,
      const AbstractBugReport& report);

  SurfaceFeatureValueCalculator(const ReportBuckets& report_buckets,
      EnumSurfaceFeatureType type);

  virtual ~SurfaceFeatureValueCalculator();
};

#endif /* SURFACEFEATUREVALUECALCULATOR_H_ */
