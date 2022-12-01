/*
 * ICSE10WithSurfaceFeatureCalculator.h
 *
 *  Created on: Dec 20, 2010
 *      Author: Chengnian SUN
 */

#ifndef _ICSE10_WITH_SURFACE_FEATURE_VECTOR_CALCULATOR_H__
#define _ICSE10_WITH_SURFACE_FEATURE_VECTOR_CALCULATOR_H__

#include "ICSE10FeatureVectorCalculator.h"

class ICSE10WithSurfaceFeatureVectorCalculator: public ICSE10FeatureVectorCalculator {
private:

protected:

  virtual vector<AbstractFeatureValueCalculator*> create_Surface_feature_vector_calculators() const;

  ICSE10WithSurfaceFeatureVectorCalculator(const ReportBuckets& report_buckets,
      const unsigned textual_feature_count,
      const unsigned surface_feature_count);

public:

  ICSE10WithSurfaceFeatureVectorCalculator(const ReportBuckets& report_buckets);

  virtual ~ICSE10WithSurfaceFeatureVectorCalculator();
};

#endif /* _ICSE10_WITH_SURFACE_FEATURE_VECTOR_CALCULATOR_H__ */
