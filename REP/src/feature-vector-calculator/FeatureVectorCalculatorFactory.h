/*
 * FeatureCalculatorFactory.h
 *
 *  Created on: Dec 21, 2010
 *      Author: Chengnian SUN
 */

#ifndef _FEATURE_VECTOR_CALCULATOR_FACTORY_H__
#define _FEATURE_VECTOR_CALCULATOR_FACTORY_H__

#include <cstdio>
#include <cassert>
#include <iostream>
#include <sstream>
#include <cstdlib>
using namespace std;

class AbstractFeatureVectorCalculator;
class ReportBuckets;

class FeatureVectorCalculatorFactory {
public:

  enum EnumFeatureCalculatorType {
    ICSE10 = 1,

    OKAPI = 2,

    ICSE10_PRUNED = 3,

    OKAPI_PRUNED = 4,

    ICSE10_WITH_SURFACE = 5,

    OKAPI_WITH_SURFACE = 6,

    RANK_NET_WITH_SURFACE = 7,

    RANKNET_OKAPI_SURFACE = 8,

    NONE = 9999
  };

  inline static EnumFeatureCalculatorType get_enum_feature_calculator_type_from_int(
      const int type_int) {
    EnumFeatureCalculatorType type =
        static_cast<EnumFeatureCalculatorType>(type_int);
    switch (type) {
    case ICSE10:
      return FeatureVectorCalculatorFactory::ICSE10;
    case OKAPI:
      return FeatureVectorCalculatorFactory::OKAPI;
    case ICSE10_PRUNED:
      return FeatureVectorCalculatorFactory::ICSE10_PRUNED;
    case OKAPI_PRUNED:
      return FeatureVectorCalculatorFactory::OKAPI_PRUNED;
    case ICSE10_WITH_SURFACE:
      return FeatureVectorCalculatorFactory::ICSE10_WITH_SURFACE;
    case OKAPI_WITH_SURFACE:
      return FeatureVectorCalculatorFactory::OKAPI_WITH_SURFACE;
    case RANK_NET_WITH_SURFACE:
      return FeatureVectorCalculatorFactory::RANK_NET_WITH_SURFACE;
    case RANKNET_OKAPI_SURFACE:
      return FeatureVectorCalculatorFactory::RANKNET_OKAPI_SURFACE;
    default:
      cerr << "invalid feature type " << type_int << endl;
      exit(1);
    }
  }

  inline static string get_enum_feature_calculator_type_mapping() {
    stringstream ss;
    ss << ICSE10 << ":ICSE10, ";
    ss << OKAPI << ":OKAPI, ";
    ss << ICSE10_PRUNED << ":ICSE10_PRUNED, ";
    ss << OKAPI_PRUNED << ":OKAPI_PRUNED, ";
    ss << ICSE10_WITH_SURFACE << ":ICSE10_WITH_SURFACE, ";
    ss << OKAPI_WITH_SURFACE << ":OKAPI_WITH_SURFACE, ";
    ss << RANK_NET_WITH_SURFACE << ":RANK_NET_WITH_SURFACE, ";
    ss << RANKNET_OKAPI_SURFACE << ":RANKNET_OKAPI_SURFACE, ";
    return ss.str();
  }

  inline static string get_enum_feature_calculator_type_string(
      const enum EnumFeatureCalculatorType type) {
    switch (type) {
    case ICSE10: {
      return "ICSE10";
    }
    case OKAPI: {
      return "OKAPI";
    }
    case ICSE10_PRUNED: {
      return "ICSE10-Pruned";
    }
    case OKAPI_PRUNED: {
      return "OKAPI-Pruned";
    }
    case ICSE10_WITH_SURFACE: {
      return "ICSE10_WITH_SURFACE";
    }
    case OKAPI_WITH_SURFACE: {
      return "OKAPI_WITH_SURFACE";
    }
    case RANK_NET_WITH_SURFACE: {
      return "RANK_NET_WITH_SURFACE";
    }
    case RANKNET_OKAPI_SURFACE: {
      return "RANKNET_OKAPI_SURFACE";
    }
    default: {
      return "NONE";
    }
    }
  }

  static AbstractFeatureVectorCalculator* create_feature_calculator(
      FILE* log_file, const ReportBuckets& report_buckets,
      const enum EnumFeatureCalculatorType feature_calculator_type);

private:

  FeatureVectorCalculatorFactory() {
  }

  virtual ~FeatureVectorCalculatorFactory() {
  }
};

#endif /* _FEATURE_VECTOR_CALCULATOR_FACTORY_H_ */
