/*
 * ComboICSE07_Similairty_Measure.h
 *
 *  Created on: Dec 17, 2010
 *      Author: Chengnian SUN
 */

#ifndef COMBOICSE07_SIMILAIRTY_MEASURE_H_
#define COMBOICSE07_SIMILAIRTY_MEASURE_H_

#include "ComboCosineSimilarityMeasure.h"

class ComboICSE07_Similairty_Measure: public ComboCosineSimilarityMeasure {

protected:

  virtual double weigh_term(const int tf, const double) {
    return 1 + log(tf) / log(2);
  }

public:

  ComboICSE07_Similairty_Measure(const int summary_weight) :
      ComboCosineSimilarityMeasure(summary_weight) {

  }

  virtual ~ComboICSE07_Similairty_Measure() {

  }
};

#endif /* COMBOICSE07_SIMILAIRTY_MEASURE_H_ */
