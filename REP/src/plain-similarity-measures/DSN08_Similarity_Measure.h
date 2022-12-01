/*
 * DSN08_Similarity_Measure.h
 *
 *  Created on: Dec 15, 2010
 *      Author: Chengnian SUN
 */

#ifndef DSN08_SIMILARITY_MEASURE_H_
#define DSN08_SIMILARITY_MEASURE_H_

#include "CosineSimilarityMeasure.h"

class DSN08_Similarity_Measure: public CosineSimilarityMeasure {

protected:
  virtual double weigh_term(const int tf, const double) {
    return 3 + 2 * (log(tf) / log(2));
  }

public:
  DSN08_Similarity_Measure(const int summary_weight) :
      CosineSimilarityMeasure(summary_weight) {
  }
  virtual ~DSN08_Similarity_Measure() {

  }
};

#endif /* DSN08_SIMILARITY_MEASURE_H_ */
