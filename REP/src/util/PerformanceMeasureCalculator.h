/*
 * PerformanceMeasureCalculator.h
 *
 *  Created on: Jan 12, 2011
 *      Author: Chengnian SUN
 */

#ifndef PERFORMANCEMEASURECALCULATOR_H_
#define PERFORMANCEMEASURECALCULATOR_H_

#include <utility>
#include <vector>

#include "../detection-model/DetectionResultOfEachDuplicate.h"

using namespace std;

class PerformanceMeasureCalculator {
public:

  static vector<std::pair<double, int> > compute_recall(const unsigned top_k,
      const vector<DetectionResultOfEachDuplicate> result_vector);

  static double compute_mean_average_precision(
      const vector<DetectionResultOfEachDuplicate> result_vector);

private:

  PerformanceMeasureCalculator();

  virtual ~PerformanceMeasureCalculator();
};

#endif /* PERFORMANCEMEASURECALCULATOR_H_ */
