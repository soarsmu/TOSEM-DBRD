/*
 * RankNetToppingAlgorithm.h
 *
 *  Created on: Dec 27, 2010
 *      Author: Chengnian SUN
 */

#ifndef _RANKNET_TOPPING_ALGORITHM_H_
#define _RANKNET_TOPPING_ALGORITHM_H_

#include <cstdio>

#include "AbstractToppingAlgorithm.h"

class AbstractRankNetLearner;
class DefaultREPParameter;

class RankNetToppingAlgorithm: public AbstractToppingAlgorithm {

private:

  AbstractRankNetLearner* m_learner;

  unsigned m_next_training_trigger;

  const static unsigned TRAINING_TRIGGER_INTERVAL = 90000;

  const DefaultREPParameter& m_default_model_parameter;

protected:

  const AbstractRankNetLearner* get_learner() const;

  void update_next_training_trigger();

  bool need_to_update() const;

  virtual void before_get_top();

  virtual double compute_similarity(const AbstractBugReport& query_report,
      const AbstractBugReport& report);

public:

  RankNetToppingAlgorithm(FILE* log_file, const ReportBuckets& buckets,
      const DefaultREPParameter& parameter) :
      AbstractToppingAlgorithm(buckets, log_file), m_learner(NULL), m_next_training_trigger(
          0), m_default_model_parameter(parameter) {
  }

  virtual ~RankNetToppingAlgorithm();

};

inline const AbstractRankNetLearner* RankNetToppingAlgorithm::get_learner() const {
  return this->m_learner;
}


#endif /* _RANKNET_TOPPING_ALGORITHM_H_ */
