/*
 * SVMToppingAlgorithm.h
 *
 *  Created on: 2010-8-4
 *      Author: Chengnian Sun.
 */

#ifndef SVMTOPPINGALGORITHM_H_
#define SVMTOPPINGALGORITHM_H_

#include <cstdio>
#include <cstdio>
using namespace std;

#include "../detection-model/ReportBuckets.h"
#include "../feature-vector-calculator/AbstractFeatureVectorCalculator.h"
#include "../feature-vector-calculator/FeatureVectorCalculatorFactory.h"
#include "../libsvm/svm.h"
#include "../topping/AbstractToppingAlgorithm.h"
#include "FeatureSpace.h"
#include "TrainingSet.h"

class SVMToppingAlgorithm: public AbstractToppingAlgorithm {
private:

  FILE* m_log_file;

private:
  // the training set.
  TrainingSet training_set;

  struct svm_model* model;

  svm_parameter parameter;

  AbstractFeatureVectorCalculator* feature_calculator;

  FeatureSpace feature_space;

  svm_node* testing_feature_vector;

  int m_probability_index;

  inline void compute_testing_feature_vector(
      const AbstractBugReport& query_report, const AbstractBugReport& report) {
    this->feature_calculator->fill_feature_values(this->testing_feature_vector,
        query_report, report);
    this->feature_space.scale_testing_vector(this->testing_feature_vector);
  }

protected:
  virtual void before_get_top();

  svm_problem* translate_svm_problem();

  inline void destroy_problem(svm_problem* problem) {
    delete[] problem->y;
    problem->y = NULL;
    delete[] problem->x;
    problem->x = NULL;
    delete problem;
  }

  virtual double compute_similarity(const AbstractBugReport& query_report,
      const AbstractBugReport& report);

  void optimize_svm_problem(svm_problem* problem);

public:

  /**
   * create the initial training set.
   */
  inline void construct_initial_training_set() {
    const ReportBuckets& buckets = this->get_buckets();
    const unsigned int master_count = buckets.get_bucket_count();
    for (unsigned int bucket_id = 0; bucket_id < master_count; bucket_id++) {
      MasterBugReport* master = buckets.get_bucket_master(bucket_id);
      const vector<const DuplicateBugReport*>& duplicates =
          master->get_duplicates();
      const unsigned int duplicate_count = duplicates.size();
      for (unsigned int dup_index = 0; dup_index < duplicate_count;
          dup_index++) {
        this->training_set.add_initial_query(duplicates[dup_index]);
      }
    }
  }

  /**
   * add a new query to the training set.
   */
  inline void add_new_query(const DuplicateBugReport& new_query) {
    this->training_set.add_new_query(new_query);
  }

  SVMToppingAlgorithm(FILE* log_file, const ReportBuckets& buckets,
      const enum FeatureVectorCalculatorFactory::EnumFeatureCalculatorType feature_calculator_type);

  virtual ~SVMToppingAlgorithm();
};

#endif /* SVMTOPPINGALGORITHM_H_ */
