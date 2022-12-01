/*
 * RankNetFeatureValueCalculator.h
 *
 *  Created on: Jan 7, 2011
 *      Author: Chengnian SUN
 */

#ifndef RANKNETFEATUREVALUECALCULATOR_H_
#define RANKNETFEATUREVALUECALCULATOR_H_

#include "AbstractFeatureValueCalculator.h"
#include <cstdio>

class DuplicateBugReport;
class AbstractBugReport;
class ReportBuckets;
class AbstractRankNetLearner;

class RankNetFeatureValueCalculator: public AbstractFeatureValueCalculator {
private:

  //	bool m_trained;
  //
  //	OkapiWrapper::BM25FParameter m_bm25f_parameter;
  AbstractRankNetLearner* m_leaner;

  FILE* m_log_file;

public:

  virtual double compute_feature_value(const AbstractBugReport& query,
      const AbstractBugReport& report);

  RankNetFeatureValueCalculator(FILE* log_file, const ReportBuckets& buckets);

  virtual ~RankNetFeatureValueCalculator();

};

#endif /* RANKNETFEATUREVALUECALCULATOR_H_ */
