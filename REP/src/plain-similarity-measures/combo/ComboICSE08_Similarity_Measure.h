/*
 * ComboICSE08_Similarity_Measure.h
 *
 *  Created on: Dec 17, 2010
 *      Author: Chengnian SUN
 */

#ifndef COMBOICSE08_SIMILARITY_MEASURE_H_
#define COMBOICSE08_SIMILARITY_MEASURE_H_

#include "ComboCosineSimilarityMeasure.h"

class ComboICSE08_Similarity_Measure: public ComboCosineSimilarityMeasure {
protected:

  virtual double weigh_term(const int tf, const double idf) {
    return tf * idf;
  }

public:
  ComboICSE08_Similarity_Measure(const int summary_weight) :
      ComboCosineSimilarityMeasure(summary_weight) {

  }

  virtual ~ComboICSE08_Similarity_Measure() {

  }
};

#endif /* COMBOICSE08_SIMILARITY_MEASURE_H_ */
