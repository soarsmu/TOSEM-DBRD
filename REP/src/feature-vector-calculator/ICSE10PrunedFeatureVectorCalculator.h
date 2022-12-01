/*
 * ICSE10PrunedFeatureCalculator.h
 *
 *  Created on: Dec 6, 2010
 *      Author: neo
 */

#ifndef _ICSE10_PRUNED_FEATURE_VECTOR_CALCULATOR_H__
#define _ICSE10_PRUNED_FEATURE_VECTOR_CALCULATOR_H__

#include "ICSE10FeatureVectorCalculator.h"

class ICSE10PrunedFeatureVectorCalculator: public ICSE10FeatureVectorCalculator {
protected:

  virtual bool accept_feature_type(
      const enum SectionType::EnumSectionType query_type,
      const enum SectionType::EnumSectionType doc_type,
      const enum IDFCollectionType::EnumIDFCollectionType idf_type) const;

  ICSE10PrunedFeatureVectorCalculator(const ReportBuckets& report_buckets,
      const unsigned textual_feature_count,
      const unsigned surface_feature_count);

public:

  ICSE10PrunedFeatureVectorCalculator(const ReportBuckets& report_buckets);

  virtual ~ICSE10PrunedFeatureVectorCalculator();
};

#endif /* _ICSE10_PRUNED_FEATURE_VECTOR_CALCULATOR_H__ */
