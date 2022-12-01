/*
 * SVMToppingAlgorithm.cpp
 *
 *  Created on: 2010-8-4
 *      Author: Chengnian Sun.
 */

#include <boost/unordered_set.hpp>
#include <cstdio>
#include <ctime>
#include <utility>

#include "../report-model/DuplicateBugReport.h"
#include "../util/MacroUtility.h"
#include "SVMToppingAlgorithm.h"

using namespace std;

#define RELEVANT_LABEL 1

#define IRRELEVANT_LABEL 0

double SVMToppingAlgorithm::compute_similarity(
    const AbstractBugReport& query_report, const AbstractBugReport& report) {
  double probability[2];

  this->compute_testing_feature_vector(query_report, report);

  svm_predict_probability(this->model, this->testing_feature_vector,
      probability);

  assert(this->m_probability_index > -1);
  assert(this->m_probability_index < 2);
  return probability[this->m_probability_index];
}

/**
 * this method is invoked before each invocation of get_top();
 *
 * Thus here we train a model here....
 */
void SVMToppingAlgorithm::before_get_top() {

  this->feature_calculator->start_to_use();

  this->training_set.populate_training_instances();

  svm_problem* problem = this->translate_svm_problem();
  if (this->model != NULL) {
    svm_destroy_model(this->model);
  }
  //time_t start_time = time(NULL);
  this->model = svm_train(problem, &(this->parameter));
  //this->training_set.destroy_problem(problem);
  this->destroy_problem(problem);
  //TRACE(cout << "training cost is " << (time(NULL) - start_time) << "\n");
  assert(2 == svm_get_nr_class(this->model));
  int labels[2];
  svm_get_labels(this->model, labels);

  for (int i = 0; i < 2; i++) {
    if (RELEVANT_LABEL == labels[i]) {
      this->m_probability_index = i;
      assert(this->m_probability_index > -1);
      break;
    }
  }
}

svm_problem* SVMToppingAlgorithm::translate_svm_problem() {
  this->feature_space.reset_validity_indicators();

  // compute feature vectors
  const vector<TrainingInstance*>& instances =
      this->training_set.get_training_instances();
  const unsigned int number_of_instances = instances.size();

  for (unsigned int index = 0; index < number_of_instances; index++) {
    TrainingInstance* instance = instances[index];

    svm_node* feature_vector = this->feature_space.get_feature_vector(index);
    this->feature_space.set_label(index, instance->get_label());
    const AbstractBugReport* query = instance->get_query_report();
    const AbstractBugReport* report = instance->get_base_report();
    this->feature_calculator->fill_feature_values(feature_vector, *query,
        *report);
  }

  this->feature_space.prune_duplicate_feature_vectors();

  svm_problem* problem = new svm_problem();
  // TODO: here to do the filtering stuff.
  pair<int, double*> label_infor =
      this->feature_space.new_and_init_label_array();
  problem->l = label_infor.first;
  problem->y = label_infor.second;
  problem->x = new svm_node*[problem->l];

  //assert(problem->l == (relevant_count + irrelevant_count));
  int number_of_relevant_instances = 0;
  int number_of_irrelevant_instances = 0;

  unsigned int vector_index = 0;
  for (int i = 0; i < problem->l; i++) {
    while (this->feature_space.is_valid_feature_vector(vector_index) == false) {
      vector_index++;
    }
    problem->x[i] = this->feature_space.get_valid_feature_vector(
        vector_index++);
    if (problem->y[i] == 0) {
      number_of_irrelevant_instances++;
    } else {
      number_of_relevant_instances++;
    }
  }

  TC_INFO(printf("R/I=%d/%d\n", number_of_relevant_instances, number_of_irrelevant_instances);)

  // scale feature vectors;
  this->feature_space.learn_scaling_parameters();
  this->feature_space.scale_feature_space();

  return problem;
}

static svm_node* create_testing_feature_vector(
    const unsigned int number_of_features,
    const unsigned int length_of_feature_vector) {
  // init testing feature vector
  svm_node* testing_feature_vector = new svm_node[length_of_feature_vector];
  for (unsigned int i = 0; i < number_of_features; i++) {
    testing_feature_vector[i].index = i;
  }
  for (unsigned int i = number_of_features; i < length_of_feature_vector; i++) {
    testing_feature_vector[i].index = -1;
  }
  return testing_feature_vector;
}

static AbstractFeatureVectorCalculator* create_feature_calculator(
    const ReportBuckets& report_buckets,
    const enum FeatureVectorCalculatorFactory::EnumFeatureCalculatorType feature_calculator_type,
    FILE* log_file) {
  return FeatureVectorCalculatorFactory::create_feature_calculator(log_file,
      report_buckets, feature_calculator_type);
}

SVMToppingAlgorithm::SVMToppingAlgorithm(FILE* log_file,
    const ReportBuckets& buckets,
    const enum FeatureVectorCalculatorFactory::EnumFeatureCalculatorType feature_calculator_type) :
    AbstractToppingAlgorithm(buckets, log_file) {
  this->training_set.set_report_buckets(buckets);

  this->parameter.svm_type = C_SVC;
  this->parameter.kernel_type = LINEAR;
  this->parameter.degree = 3;
  this->parameter.gamma = 0;
  this->parameter.coef0 = 0;
  this->parameter.nu = 0.5;
  this->parameter.cache_size = 800;
  this->parameter.C = 1;
  this->parameter.eps = 1e-3;
  this->parameter.p = 0.1;
  this->parameter.shrinking = 1;
  this->parameter.probability = 1;
  this->parameter.nr_weight = 0;
  this->parameter.weight_label = NULL;
  this->parameter.weight = NULL;
  this->model = NULL;

  assert(this->m_log_file == log_file);
  this->feature_calculator = create_feature_calculator(buckets,
      feature_calculator_type, log_file);

  const unsigned int number_of_features =
      this->feature_calculator->get_feature_count();
  this->feature_space.init(number_of_features);

  this->testing_feature_vector = create_testing_feature_vector(
      number_of_features, this->feature_space.get_length_of_feature_vector());

  this->m_probability_index = -1;
}

SVMToppingAlgorithm::~SVMToppingAlgorithm() {
  if (this->model != NULL) {
    svm_destroy_model(this->model);
    this->model = NULL;
  }
  if (this->testing_feature_vector != NULL) {
    delete[] this->testing_feature_vector;
    this->testing_feature_vector = NULL;
  }
  if (this->feature_calculator != NULL) {
    delete this->feature_calculator;
    this->feature_calculator = NULL;
  }
}
