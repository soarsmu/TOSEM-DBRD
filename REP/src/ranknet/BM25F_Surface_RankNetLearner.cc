/*
 * BM25F_Surface_RankNetLearner.cc
 *
 *  Created on: Jan 10, 2011
 *      Author: Chengnian SUN
 */

#include "BM25F_Surface_RankNetLearner.h"
#include "../feature-value-calculator/SurfaceFeatureValueCalculator.h"
#include "DefaultREPParameter.h"
#include "../util/MacroUtility.h"

void BM25F_Surface_RankNetLearner::find_a_better_model() {
  BM25FRankNetLearner::find_a_better_model();
  this->m_best_surface_weight = this->m_surface_weight;
}

float BM25F_Surface_RankNetLearner::compute_similarity_with_textual_sim(
    const float textual_similarity, const AbstractBugReport& query,
    const AbstractBugReport& base) const {
  return textual_similarity + this->compute_categorial_similarity(query, base);
}

void BM25F_Surface_RankNetLearner::learning_done() {
  BM25FRankNetLearner::learning_done();
  this->m_surface_weight = this->m_best_surface_weight;
  this->m_surface_weight.print(this->get_log_file());
}

BM25F_Surface_RankNetLearner::BM25F_Surface_RankNetLearner(FILE* log_file,
    const ReportBuckets& report_buckets, const DefaultREPParameter& parameter) :
    BM25FRankNetLearner(log_file, report_buckets, parameter) {
}

void BM25F_Surface_RankNetLearner::initialize_model_parameters(
    const unsigned round) {
  BM25FRankNetLearner::initialize_model_parameters(round);

  switch (round) {
  case 1: {
    this->m_surface_weight.initialize(
        this->m_default_model_parameter.get_component_weight(),
        this->m_default_model_parameter.is_component_weight_fixed(),
        this->m_default_model_parameter.get_sub_component_weight(),
        this->m_default_model_parameter.is_sub_component_weight_fixed(),
        this->m_default_model_parameter.get_report_type_weight(),
        this->m_default_model_parameter.is_report_type_weight_fixed(),
        this->m_default_model_parameter.get_priority_weight(),
        this->m_default_model_parameter.is_priority_weight_fixed(),
        this->m_default_model_parameter.get_version_weight(),
        this->m_default_model_parameter.is_version_weight_fixed());
    break;
  }
  case 2: {
    break;
  }
  default:
    ERROR_HERE("illegal round");
    break;
  }
}

void BM25F_Surface_RankNetLearner::before_tune_parameters_on_one_pair() {
  BM25FRankNetLearner::before_tune_parameters_on_one_pair();
  this->m_tuning_surface_weight = this->m_surface_weight;
}

void BM25F_Surface_RankNetLearner::after_tune_parameters_on_one_pair() {
  BM25FRankNetLearner::after_tune_parameters_on_one_pair();
  this->m_surface_weight = this->m_tuning_surface_weight;
}

void BM25F_Surface_RankNetLearner::tune_parameters_on_one_pair(
    const RankNetTrainingInstance& training_pair, const double learning_rate) {
  /*
   * evaluate the gradient.
   *
   * 1) compute the RNC cost.
   */
  const AbstractBugReport& query_report = training_pair.get_query();
  const AbstractBugReport& relevant_report =
      training_pair.get_relevant_report();
  const AbstractBugReport& irrelevant_report =
      training_pair.get_irrelevant_report();

  /*
   * 2) compute and perform the gradient.
   *
   * RNC(Y) = log(1 + exp(Y)) where Y = y2 - y1, y2 and y1 are the similarity scores.
   *
   * RNC'(Y) = exp(Y) / (1 + exp(Y))
   *
   */
  const double derivative_of_rnc_wrt_Y = this->compute_rnc_derivative_wrt_Y(
      training_pair);

  BM25FRankNetLearner::tune_parameters_on_one_pair(training_pair,
      learning_rate);

  if (!this->m_surface_weight.is_component_weight_fixed()) {
    // derivative wrt. component-weight
    const double derivative_component_weight =
        this->compute_rnc_derivative_wrt_non_textual_weight(
            derivative_of_rnc_wrt_Y, query_report, relevant_report,
            irrelevant_report,
            SurfaceFeatureValueCalculator::compute_component_similarity);
    this->m_tuning_surface_weight.increase_component_weight(
        -learning_rate * derivative_component_weight);
  }

  if (!this->m_surface_weight.is_sub_component_weight_fixed()) {
    // derivative wrt. component-weight
    const double derivative_sub_component_weight =
        this->compute_rnc_derivative_wrt_non_textual_weight(
            derivative_of_rnc_wrt_Y, query_report, relevant_report,
            irrelevant_report,
            SurfaceFeatureValueCalculator::compute_sub_component_similarity);
    this->m_tuning_surface_weight.increase_sub_component_weight(
        -learning_rate * derivative_sub_component_weight);
  }

  if (!this->m_surface_weight.is_report_type_weight_fixed()) {
    // derivative wrt. component-weight
    const double derivative_report_type_weight =
        this->compute_rnc_derivative_wrt_non_textual_weight(
            derivative_of_rnc_wrt_Y, query_report, relevant_report,
            irrelevant_report,
            SurfaceFeatureValueCalculator::compute_report_type_similarity);
    this->m_tuning_surface_weight.increase_report_type_weight(
        -learning_rate * derivative_report_type_weight);

  }

  if (!this->m_surface_weight.is_priority_weight_fixed()) {
    // derivative wrt. component-weight
    const double derivative_priority_weight =
        this->compute_rnc_derivative_wrt_non_textual_weight(
            derivative_of_rnc_wrt_Y, query_report, relevant_report,
            irrelevant_report,
            SurfaceFeatureValueCalculator::compute_priority_similarity);
    this->m_tuning_surface_weight.increase_priority_weight(
        -learning_rate * derivative_priority_weight);
  }

  if (!this->m_surface_weight.is_version_weight_fixed()) {
    // derivative wrt. component-weight
    const double derivative_version_weight =
        this->compute_rnc_derivative_wrt_non_textual_weight(
            derivative_of_rnc_wrt_Y, query_report, relevant_report,
            irrelevant_report,
            SurfaceFeatureValueCalculator::compute_version_similarity);
    this->m_tuning_surface_weight.increase_version_weight(
        -learning_rate * derivative_version_weight);

  }

}

float BM25F_Surface_RankNetLearner::compute_categorial_similarity(
    const AbstractBugReport& query_report,
    const AbstractBugReport& base_report) const {
  const float comp_sim = this->m_surface_weight.get_component_weight()
      * SurfaceFeatureValueCalculator::compute_component_similarity(
          query_report, base_report);

  assert(comp_sim == comp_sim);

  const float sub_comp_sim = this->m_surface_weight.get_sub_component_weight()
      * SurfaceFeatureValueCalculator::compute_sub_component_similarity(
          query_report, base_report);

  assert(sub_comp_sim == sub_comp_sim);

  const float report_type_sim = this->m_surface_weight.get_report_type_weight()
      * SurfaceFeatureValueCalculator::compute_report_type_similarity(
          query_report, base_report);

  assert(report_type_sim == report_type_sim);

  const float priority_sim = this->m_surface_weight.get_priority_weight()
      * SurfaceFeatureValueCalculator::compute_priority_similarity(query_report,
          base_report);

  assert(priority_sim == priority_sim);

  const float version_sim = this->m_surface_weight.get_version_weight()
      * SurfaceFeatureValueCalculator::compute_version_similarity(query_report,
          base_report);

  assert(version_sim == version_sim);

  return comp_sim + sub_comp_sim + report_type_sim + priority_sim + version_sim;
}

float BM25F_Surface_RankNetLearner::compute_similarity(
    const AbstractBugReport& query_report,
    const AbstractBugReport& base_report) const {

  const float textual_similarity = BM25FRankNetLearner::compute_similarity(
      query_report, base_report);

  assert(textual_similarity == textual_similarity);
  //	const double bigram_text_sim = this->m_surface_weight.get_bigram_weight() * OkapiWrapper::compute_bm25f();

//  const double comp_sim = this->m_surface_weight.get_component_weight()
//      * SurfaceFeatureValueCalculator::compute_component_similarity(
//          query_report, base_report);
//
//  assert(comp_sim == comp_sim);
//
//  const double sub_comp_sim = this->m_surface_weight.get_sub_component_weight()
//      * SurfaceFeatureValueCalculator::compute_sub_component_similarity(
//          query_report, base_report);
//
//  assert(sub_comp_sim == sub_comp_sim);
//
//  const double report_type_sim = this->m_surface_weight.get_report_type_weight()
//      * SurfaceFeatureValueCalculator::compute_report_type_similarity(
//          query_report, base_report);
//
//  assert(report_type_sim == report_type_sim);
//
//  const double priority_sim = this->m_surface_weight.get_priority_weight()
//      * SurfaceFeatureValueCalculator::compute_priority_similarity(query_report,
//          base_report);
//
//  assert(priority_sim == priority_sim);
//
//  const double version_sim = this->m_surface_weight.get_version_weight()
//      * SurfaceFeatureValueCalculator::compute_version_similarity(query_report,
//          base_report);
//
//  assert(version_sim == version_sim);

//  return textual_similarity + comp_sim + sub_comp_sim + report_type_sim
//      + priority_sim + version_sim;
  return compute_similarity_with_textual_sim(textual_similarity, query_report,
      base_report);

}

BM25F_Surface_RankNetLearner::~BM25F_Surface_RankNetLearner() {
}
