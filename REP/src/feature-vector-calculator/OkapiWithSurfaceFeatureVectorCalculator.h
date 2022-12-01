/*
 * OkapiWithSurfaceFeatureCalculator.h
 *
 *  Created on: Dec 21, 2010
 *      Author: Chengnian SUN
 */

#ifndef _OKAPI_WITH_SURFACE_FEATURE_VECTOR_CALCULATOR_H__
#define _OKAPI_WITH_SURFACE_FEATURE_VECTOR_CALCULATOR_H__

#include "OkapiFeatureVectorCalculator.h"

class OkapiWithSurfaceFeatureVectorCalculator: public OkapiFeatureVectorCalculator {

protected:

  virtual vector<AbstractFeatureValueCalculator*> create_Surface_feature_vector_calculators() const;

  OkapiWithSurfaceFeatureVectorCalculator(const ReportBuckets& report_buckets,
      const unsigned textual_feature_count,
      const unsigned surface_feature_count);
public:

  OkapiWithSurfaceFeatureVectorCalculator(const ReportBuckets& report_buckets);

  virtual ~OkapiWithSurfaceFeatureVectorCalculator();
};

#endif /* _OKAPI_WITH_SURFACE_FEATURE_VECTOR_CALCULATOR_H__ */
