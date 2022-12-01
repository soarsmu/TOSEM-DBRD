/*
 * PerformanceMeasureCalculator.cc
 *
 *  Created on: Jan 12, 2011
 *      Author: Chengnian SUN
 */

#include "PerformanceMeasureCalculator.h"

double PerformanceMeasureCalculator::compute_mean_average_precision(
    const vector<DetectionResultOfEachDuplicate> detection_result_vector) {
  const unsigned number_of_duplicates = detection_result_vector.size();
  double average_precisions = 0;
  for (unsigned int i = 0; i < number_of_duplicates; i++) {
    const DetectionResultOfEachDuplicate& result = detection_result_vector[i];
    const unsigned index = result.get_index_where_master_detected();
    if (index
        == DetectionResultOfEachDuplicate::INVALID_INDEX_OF_MASTER_DETECTED) {
      average_precisions += 0;
    } else {
      average_precisions += 1.0 / (index + 1);
    }
  }
  return average_precisions / number_of_duplicates;
}

vector<std::pair<double, int> > PerformanceMeasureCalculator::compute_recall(
    const unsigned top_k,
    const vector<DetectionResultOfEachDuplicate> detection_result_vector) {

  vector<int> correct_result_list(top_k, 0);

  const unsigned number_of_duplicates = detection_result_vector.size();

  for (unsigned int i = 0; i < number_of_duplicates; i++) {
    const DetectionResultOfEachDuplicate& result = detection_result_vector[i];
    for (unsigned j = 0; j < top_k; j++) {
      if (j >= result.get_index_where_master_detected()) {
        correct_result_list[j]++;
      }
    }
  }

  vector<std::pair<double, int> > result;
  for (unsigned i = 0; i < top_k; i++) {
    const int value = correct_result_list[i];
    result.push_back(
        std::make_pair(static_cast<double>(value) / number_of_duplicates,
            value));
  }

  return result;
  }

PerformanceMeasureCalculator::PerformanceMeasureCalculator() {

}

PerformanceMeasureCalculator::~PerformanceMeasureCalculator() {
}
