/*
 * RankNetToppingAlgorithm.cc
 *
 *  Created on: Dec 27, 2010
 *      Author: Chengnian SUN
 */
#include "RankNetToppingAlgorithm.h"

#include <cassert>
#include <vector>
#include <cmath>
#include <cstdio>
using namespace std;

#include "../detection-model/ReportBuckets.h"
#include "../ranknet/BM25F_Surface_RankNetLearner.h"

class MasterBugReport;
class DuplicateBugReport;

void RankNetToppingAlgorithm::update_next_training_trigger() {
  this->m_next_training_trigger = this->get_buckets().get_bucket_count()
      + TRAINING_TRIGGER_INTERVAL;
}

bool RankNetToppingAlgorithm::need_to_update() const {
  return this->m_next_training_trigger <= this->get_buckets().get_bucket_count();
}

/**
 * create the training set and tune the parameters.
 */
void RankNetToppingAlgorithm::before_get_top() {
  if (this->m_learner == NULL) {
    /*
     * 1) create a training set
     */
    this->m_learner = new BM25F_Surface_RankNetLearner(this->m_log_file,
        this->get_buckets(), this->m_default_model_parameter);

    /*
     * 2) tune the parameters
     */
    this->m_learner->learn();
    //		/*
    //		 * 3) dispose the learner
    //		 */
      this->update_next_training_trigger();

  } else if (this->need_to_update()) {
    this->m_learner->learn();
    this->update_next_training_trigger();

  }
}

double RankNetToppingAlgorithm::compute_similarity(
    const AbstractBugReport& query_report, const AbstractBugReport& report) {
  assert(this->m_learner);
  return this->m_learner->compute_similarity(query_report, report);
}

RankNetToppingAlgorithm::~RankNetToppingAlgorithm() {
  if (this->m_learner != NULL) {
    delete this->m_learner;
  }
}
