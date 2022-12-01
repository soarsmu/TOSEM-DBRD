/*
 * ComboDSN08_Similarity_Measure.h
 *
 *  Created on: Dec 17, 2010
 *      Author: Chengnian SUN
 */

#ifndef COMBODSN08_SIMILARITY_MEASURE_H_
#define COMBODSN08_SIMILARITY_MEASURE_H_

#include "ComboCosineSimilarityMeasure.h"

class ComboDSN08_Similarity_Measure: public ComboCosineSimilarityMeasure {
protected:
  virtual double weigh_term(const int tf, const double) {
    return 3 + 2 * (log(tf) / log(2));
  }

public:
  ComboDSN08_Similarity_Measure(const int summary_weight) :
      ComboCosineSimilarityMeasure(summary_weight) {

  }

  virtual ~ComboDSN08_Similarity_Measure() {

  }
};

#endif /* COMBODSN08_SIMILARITY_MEASURE_H_ */
