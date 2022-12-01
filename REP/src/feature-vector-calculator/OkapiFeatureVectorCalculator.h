#pragma once

#ifndef _OKAPI_FEATURE_VECTOR_CALCULATOR_H__
#define _OKAPI_FEATURE_VECTOR_CALCULATOR_H__

#include "ICSE10FeatureVectorCalculator.h"
#include <vector>
using namespace std;

class OkapiFeatureVectorCalculator: public ICSE10FeatureVectorCalculator {

protected:

  virtual AbstractFeatureValueCalculator* create_feature_value_calculator(
      const enum SectionType::EnumSectionType query_type,
      const enum SectionType::EnumSectionType doc_type,
      const enum IDFCollectionType::EnumIDFCollectionType idf_type) const;

  OkapiFeatureVectorCalculator(const ReportBuckets& report_buckets,
      const unsigned textual_feature_count,
      const unsigned surface_feature_count);

public:

  OkapiFeatureVectorCalculator(const ReportBuckets& report_buckets);

  virtual ~OkapiFeatureVectorCalculator(void);
};

#endif /*_OKAPI_FEATURE_VECTOR_CALCULATOR_H__*/
