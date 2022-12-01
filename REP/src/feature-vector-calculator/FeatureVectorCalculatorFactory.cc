/*
 * FeatureVectorCalculatorFactory.cc
 *
 *  Created on: Feb 1, 2011
 *      Author: neo
 */
#include "FeatureVectorCalculatorFactory.h"

#include "ICSE10FeatureVectorCalculator.h"
#include "ICSE10PrunedFeatureVectorCalculator.h"
#include "ICSE10WithSurfaceFeatureVectorCalculator.h"
#include "OkapiPrunedFeatureVectorCalculator.h"
#include "OkapiFeatureVectorCalculator.h"
#include "AbstractFeatureVectorCalculator.h"
#include "OkapiWithSurfaceFeatureVectorCalculator.h"
#include "RankNetWithSurfaceFeactureVectorCalculator.h"
#include "RanknetOkapiSurfaceFeatureVectorCalculator.h"

AbstractFeatureVectorCalculator* FeatureVectorCalculatorFactory::create_feature_calculator(
    FILE* log_file, const ReportBuckets& report_buckets,
    const enum EnumFeatureCalculatorType feature_calculator_type) {
  AbstractFeatureVectorCalculator* calculator;

  switch (feature_calculator_type) {
  case ICSE10: {
    calculator = new ICSE10FeatureVectorCalculator(report_buckets);
    break;
  }
  case OKAPI: {
    calculator = new OkapiFeatureVectorCalculator(report_buckets);
    break;
  }
  case ICSE10_PRUNED: {
    calculator = new ICSE10PrunedFeatureVectorCalculator(report_buckets);
    break;
  }
  case OKAPI_PRUNED: {
    calculator = new OkapiPrunedFeatureVectorCalculator(report_buckets);
    break;
  }
  case ICSE10_WITH_SURFACE: {
    calculator = new ICSE10WithSurfaceFeatureVectorCalculator(report_buckets);
    break;
  }
  case OKAPI_WITH_SURFACE: {
    calculator = new OkapiWithSurfaceFeatureVectorCalculator(report_buckets);
    break;
  }
  case RANK_NET_WITH_SURFACE: {
    calculator = new RankNetWithSurfaceFeactureVectorCalculator(log_file,
        report_buckets);
    break;
  }
  case RANKNET_OKAPI_SURFACE: {
    calculator = new RanknetOkapiSurfaceFeatureVectorCalculator(log_file,
        report_buckets);
    break;
  }
  default: {
    calculator = NULL;
    assert(false);
    fprintf(stderr, "ERROR: un-handled feature calculator type %d, at %d %s\n",
        feature_calculator_type, __LINE__, __FILE__);
    exit(1);
  }
  }
  calculator->init_feature_vector_calculator();
  return calculator;
}
