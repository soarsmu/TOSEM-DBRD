/*
 * RankNetWithSurfaceFeactureVectorCalculator.h
 *
 *  Created on: Jan 7, 2011
 *      Author: Chengnian SUN
 */

#ifndef RANKNETWITHSURFACEFEACTUREVECTORCALCULATOR_H_
#define RANKNETWITHSURFACEFEACTUREVECTORCALCULATOR_H_
#include <cstdio>
#include "AbstractFeatureVectorCalculator.h"

class RankNetWithSurfaceFeactureVectorCalculator: public AbstractFeatureVectorCalculator {
private:
  FILE* m_log_file;

protected:

  virtual vector<AbstractFeatureValueCalculator*> create_Textual_feature_vector_calculators() const;

  virtual vector<AbstractFeatureValueCalculator*> create_Surface_feature_vector_calculators() const;

  RankNetWithSurfaceFeactureVectorCalculator(FILE* log_file,
      const ReportBuckets& report_buckets, const unsigned textual_feature_count,
      const unsigned surface_feature_count);

public:

  RankNetWithSurfaceFeactureVectorCalculator(FILE* log_file,
      const ReportBuckets& report_buckets);

  virtual ~RankNetWithSurfaceFeactureVectorCalculator();

};

#endif /* RANKNETWITHSURFACEFEACTUREVECTORCALCULATOR_H_ */
