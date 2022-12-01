/*
 * OkapiPrunedFeatureCalculator.h
 *
 *  Created on: Dec 7, 2010
 *      Author: Chengnian SUN
 */

#ifndef _OKAPI_PRUNED_FEATURE_VECTOR_CALCULATOR_H__
#define _OKAPI_PRUNED_FEATURE_VECTOR_CALCULATOR_H__

#include "OkapiFeatureVectorCalculator.h"

class OkapiPrunedFeatureVectorCalculator: public OkapiFeatureVectorCalculator {
protected:

  virtual bool accept_feature_type(
      const enum SectionType::EnumSectionType query_type,
      const enum SectionType::EnumSectionType doc_type,
      const enum IDFCollectionType::EnumIDFCollectionType idf_type) const;

  OkapiPrunedFeatureVectorCalculator(const ReportBuckets& report_buckets,
      const unsigned textual_feature_count,
      const unsigned surface_feature_count);

public:

  OkapiPrunedFeatureVectorCalculator(const ReportBuckets& report_buckets);

  virtual ~OkapiPrunedFeatureVectorCalculator();
};

#endif /* _OKAPI_PRUNED_FEATURE_VECTOR_CALCULATOR_H__ */
