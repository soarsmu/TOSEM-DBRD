/*
 * RanknetOkapiSurfaceFeatureVectorCalculator.h
 *
 *  Created on: Jan 7, 2011
 *      Author: Chengnian SUN
 */

#ifndef RANKNETOKAPISURFACEFEATUREVECTORCALCULATOR_H_
#define RANKNETOKAPISURFACEFEATUREVECTORCALCULATOR_H_

#include "OkapiWithSurfaceFeatureVectorCalculator.h"
#include <cstdio>

class RanknetOkapiSurfaceFeatureVectorCalculator: public OkapiWithSurfaceFeatureVectorCalculator {
private:
  FILE* m_log_file;

protected:

  virtual vector<AbstractFeatureValueCalculator*> create_Textual_feature_vector_calculators() const;

  RanknetOkapiSurfaceFeatureVectorCalculator(FILE* log_file,
      const ReportBuckets& report_buckets, const unsigned textual_feature_count,
      const unsigned surface_feature_count);

public:

  RanknetOkapiSurfaceFeatureVectorCalculator(FILE* log_file,
      const ReportBuckets& report_buckets);

  virtual ~RanknetOkapiSurfaceFeatureVectorCalculator();

};

#endif /* RANKNETOKAPISURFACEFEATUREVECTORCALCULATOR_H_ */
