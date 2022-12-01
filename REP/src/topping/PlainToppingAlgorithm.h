/*
 * PlainToppingAlgorithm.h
 *
 *  Created on: Dec 10, 2010
 *      Author: Chengnian SUN
 */

#ifndef PLAINTOPPINGALGORITHM_H_
#define PLAINTOPPINGALGORITHM_H_

#include "AbstractToppingAlgorithm.h"
#include "../plain-similarity-measures/PlainSimilarityMeasureFactory.h"

class IPlainSimilarityMeasure;

class PlainToppingAlgorithm: public AbstractToppingAlgorithm {
private:
  IPlainSimilarityMeasure* m_similarity_measure;

protected:

  virtual void before_get_top();

  virtual double compute_similarity(const AbstractBugReport& query_report,
      const AbstractBugReport& report);

public:
  inline PlainToppingAlgorithm(const ReportBuckets& buckets,
      const enum PlainSimilarityMeasureFactory::SimilarityMeasureType similarity_measure_type,  FILE* log_file) :
      AbstractToppingAlgorithm(buckets,log_file), m_similarity_measure(
          PlainSimilarityMeasureFactory::create_similarity_measure(
              similarity_measure_type)) {

  }
  virtual ~PlainToppingAlgorithm();
};

#endif /* PLAINTOPPINGALGORITHM_H_ */
