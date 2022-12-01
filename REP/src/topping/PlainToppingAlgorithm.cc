/*
 * PlainToppingAlgorithm.cpp
 *
 *  Created on: Dec 10, 2010
 *      Author: Chengnian SUN
 */

#include "PlainToppingAlgorithm.h"
#include "../plain-similarity-measures/IPlainSimilarityMeasure.h"

double PlainToppingAlgorithm::compute_similarity(
    const AbstractBugReport& query_report, const AbstractBugReport& report) {
  return this->m_similarity_measure->compute_similarity(query_report, report,
      this->get_buckets());
}

void PlainToppingAlgorithm::before_get_top() {
}

PlainToppingAlgorithm::~PlainToppingAlgorithm() {
  delete this->m_similarity_measure;
}

