/*
 * ICSE07_W1.h
 *
 *  Created on: Dec 11, 2010
 *      Author: Chengnian SUN
 */

#ifndef _ICSE07_Similarity_Measure_
#define _ICSE07_Similarity_Measure_

#include "IPlainSimilarityMeasure.h"
#include "CosineSimilarityMeasure.h"
#include <boost/unordered_map.hpp>
#include <cmath>
using namespace std;
using namespace boost;

class ICSE07_Similarity_Measure: public CosineSimilarityMeasure {
private:

protected:

  virtual double weigh_term(const int tf, const double) {
    return 1 + log(tf) / log(2);
  }

public:

  ICSE07_Similarity_Measure(const int summary_weight) :
      CosineSimilarityMeasure(summary_weight) {
  }

};

#endif /* _ICSE07_Similarity_Measure_ */
