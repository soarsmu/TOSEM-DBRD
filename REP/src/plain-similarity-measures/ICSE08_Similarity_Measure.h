/*
 * ICSE08_Similarity_Measure.h
 *
 *  Created on: Dec 15, 2010
 *      Author: Chengnian SUN
 */

#ifndef ICSE08_SIMILARITY_MEASURE_H_
#define ICSE08_SIMILARITY_MEASURE_H_

#include "CosineSimilarityMeasure.h"

class ICSE08_Similarity_Measure: public CosineSimilarityMeasure {
protected:

  virtual double weigh_term(const int tf, const double idf) {
    return tf * idf;
  }

public:

  ICSE08_Similarity_Measure(const int summary_weight) :
      CosineSimilarityMeasure(summary_weight) {

  }

  virtual ~ICSE08_Similarity_Measure() {

  }
};

#endif /* ICSE08_SIMILARITY_MEASURE_H_ */
