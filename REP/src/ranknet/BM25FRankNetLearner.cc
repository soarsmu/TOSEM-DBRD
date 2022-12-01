/*
 * RankNetLearner.cc
 *
 *  Created on: Jan 4, 2011
 *      Author: Chengnian SUN
 */

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>

#include "../detection-model/ReportBuckets.h"
#include "../report-model/AbstractBugReport.h"
#include "../util/MacroUtility.h"
#include "BM25FRankNetLearner.h"
#include "DefaultREPParameter.h"

using namespace std;

void BM25FRankNetLearner::learn() {

}

void BM25FRankNetLearner::find_a_better_model() {
  this->m_best_Unigram_bm25f_parameter = this->m_Unigram_bm25f_parameter;
  this->m_best_Bigram_bm25f_parameter = this->m_Bigram_bm25f_parameter;
}

float BM25FRankNetLearner::compute_similarity_with_textual_sim(
    const float textual_similarity, const AbstractBugReport&,
    const AbstractBugReport&) const {
  return textual_similarity;
}

float BM25FRankNetLearner::weigh_textual_similarity(
    const float textual_similarity) const {
  return textual_similarity * this->m_Unigram_bm25f_parameter.get_total_weight();
}

void BM25FRankNetLearner::tune_gram_parameters_on_one_pair(
    const double learning_rate, const double derivative_of_rnc_wrt_Y,
    IDFCollection* idf_collection,
    OkapiWrapper::AverageLengthInfo average_length_info,
    const StructuredSection& query_section,
    const StructuredSection& relevant_section,
    const StructuredSection& irrelevant_section,
    const BM25FParameter& currentBm25fparameter,
    BM25FParameter& parameterToTune) {

  assert(&currentBm25fparameter != &parameterToTune);
  // FIXME! you cannot tuning parameters while using it. you should make a copy first, and use the copy to compute derivatives.
  if (!currentBm25fparameter.is_total_weight_fixed()) {
    const double derivative = this->compute_rnc_derivative_wrt_textual_weight(
        derivative_of_rnc_wrt_Y, idf_collection, average_length_info,
        query_section, relevant_section, irrelevant_section,
        currentBm25fparameter);
    parameterToTune.increase_total_weight(-learning_rate * derivative);
  }

  const double new_derivative_of_rnc_wrt_Y = derivative_of_rnc_wrt_Y
      * currentBm25fparameter.get_total_weight();
  // derivative wrt. k1
  if (!currentBm25fparameter.is_k1_fixed()) {
    const double derivative_k1 = this->compute_rnc_derivative_wrt_k1(
        new_derivative_of_rnc_wrt_Y, idf_collection, average_length_info,
        query_section, relevant_section, irrelevant_section,
        currentBm25fparameter);
    //	bm25parameter.k1 -= learning_rate * derivative_k1;
    parameterToTune.increase_k1(-learning_rate * derivative_k1);
  }

  if (!currentBm25fparameter.is_summary_weight_fixed()) {
    // derivative with respect to summary_weight
    const double derivative_summary_weight =
        this->compute_rnc_derivative_wrt_summary_weight(
            new_derivative_of_rnc_wrt_Y, idf_collection, average_length_info,
            query_section, relevant_section, irrelevant_section,
            currentBm25fparameter);
    //	bm25parameter.summary_weight = learning_rate * derivative_summary_weight;
    parameterToTune.increase_summary_weight(
        -learning_rate * derivative_summary_weight);
  }

  if (!currentBm25fparameter.is_summary_b_fixed()) {
    // derivative with respect to summary_b
    const double derivative_summary_b =
        this->compute_rnc_derivative_wrt_summary_b(new_derivative_of_rnc_wrt_Y,
            idf_collection, average_length_info, query_section,
            relevant_section, irrelevant_section, currentBm25fparameter);
    //	bm25parameter.summary_b = learning_rate * derivative_summary_b;
    parameterToTune.increase_summary_b(-learning_rate * derivative_summary_b);
  }

  if (!currentBm25fparameter.is_description_b_fixed()) {
    // derivative with respect to description_b'
    const double derivative_description_b =
        this->compute_rnc_derivative_wrt_description_b(
            new_derivative_of_rnc_wrt_Y, idf_collection, average_length_info,
            query_section, relevant_section, irrelevant_section,
            currentBm25fparameter);
    //	bm25parameter.description_b = learning_rate * derivative_description_b;
    parameterToTune.increase_description_b(
        -learning_rate * derivative_description_b);
  }

  if (!currentBm25fparameter.is_description_weight_fixed()) {
    // derivative with respect to summary_weight
    const double derivative_description_weight =
        this->compute_rnc_derivative_wrt_description_weight(
            new_derivative_of_rnc_wrt_Y, idf_collection, average_length_info,
            query_section, relevant_section, irrelevant_section,
            currentBm25fparameter);
    //	bm25parameter.summary_weight = learning_rate * derivative_summary_weight;
    parameterToTune.increase_description_weight(
        -learning_rate * derivative_description_weight);
  }

  if (!currentBm25fparameter.is_k3_fixed()) {
    const double derivative_k3 = this->compute_rnc_derivative_wrt_k3(
        new_derivative_of_rnc_wrt_Y, idf_collection, average_length_info,
        query_section, relevant_section, irrelevant_section,
        currentBm25fparameter);
    parameterToTune.increase_k3(-learning_rate * derivative_k3);
  }
}

void BM25FRankNetLearner::before_tune_parameters_on_one_pair() {
  this->m_tuning_Unigram_bm25f_parameter = this->m_Unigram_bm25f_parameter;
  this->m_tuning_Bigram_bm25f_parameter = this->m_Bigram_bm25f_parameter;
}

void BM25FRankNetLearner::after_tune_parameters_on_one_pair() {
  this->m_Unigram_bm25f_parameter = this->m_tuning_Unigram_bm25f_parameter;
  this->m_Bigram_bm25f_parameter = this->m_tuning_Bigram_bm25f_parameter;
}

double BM25FRankNetLearner::compute_rnc_derivative_wrt_k3(
    const double rnc_derivative_wrt_Y, IDFCollection* idf_collection,
    const OkapiWrapper::AverageLengthInfo& average_length_info,
    const StructuredSection& query_section,
    const StructuredSection& relevant_section,
    const StructuredSection& irrelevant_section,
    const BM25FParameter& bm25f_parameter) const {
  const double irrelevant = OkapiWrapper::compute_bf25f_derivative_wrt_k3(
      idf_collection, average_length_info, query_section, irrelevant_section,
      bm25f_parameter);
  const double relevant = OkapiWrapper::compute_bf25f_derivative_wrt_k3(
      idf_collection, average_length_info, query_section, relevant_section,
      bm25f_parameter);
  return rnc_derivative_wrt_Y * (irrelevant - relevant);
}

double BM25FRankNetLearner::compute_rnc_derivative_wrt_description_b(
    const double rnc_derivative_wrt_Y, IDFCollection* idf_collection,
    const OkapiWrapper::AverageLengthInfo& average_length_info,
    const StructuredSection& query_section,
    const StructuredSection& relevant_section,
    const StructuredSection& irrelevant_section,
    const BM25FParameter& bm25f_parameter) const {

  const double derivative_of_bm25f_wrt_description_b_irrelevant =
      OkapiWrapper::compute_bf25f_derivative_wrt_description_b(idf_collection,
          average_length_info, query_section, irrelevant_section,
          bm25f_parameter);

  const double derivative_of_bm25f_wrt_description_b_relevant =
      OkapiWrapper::compute_bf25f_derivative_wrt_description_b(idf_collection,
          average_length_info, query_section, relevant_section,
          bm25f_parameter);

  return rnc_derivative_wrt_Y
      * (derivative_of_bm25f_wrt_description_b_irrelevant
          - derivative_of_bm25f_wrt_description_b_relevant);
}

double BM25FRankNetLearner::compute_rnc_derivative_wrt_textual_weight(
    const double rnc_derivative_wrt_Y, IDFCollection* idf_collection,
    const OkapiWrapper::AverageLengthInfo& average_length_info,
    const StructuredSection& query_section,
    const StructuredSection& relevant_section,
    const StructuredSection& irrelevant_section,
    const BM25FParameter& bm25f_parameter) const {

  const double derivative_irrelevant = OkapiWrapper::compute_bm25f(
      idf_collection, average_length_info, query_section, irrelevant_section,
      bm25f_parameter);

  const double derivative_relevant = OkapiWrapper::compute_bm25f(idf_collection,
      average_length_info, query_section, relevant_section, bm25f_parameter);

  return rnc_derivative_wrt_Y * (derivative_irrelevant - derivative_relevant);
}

double BM25FRankNetLearner::compute_rnc_derivative_wrt_summary_b(
    const double rnc_derivative_wrt_Y, IDFCollection* idf_collection,
    const OkapiWrapper::AverageLengthInfo& average_length_info,
    const StructuredSection& query_section,
    const StructuredSection& relevant_section,
    const StructuredSection& irrelevant_section,
    const BM25FParameter& bm25f_parameter) const {

  const double derivative_of_bm25f_wrt_summary_b_irrelevant =
      OkapiWrapper::compute_bf25f_derivative_wrt_summary_b(idf_collection,
          average_length_info, query_section, irrelevant_section,
          bm25f_parameter);

  const double derivative_of_bm25f_wrt_summary_b_relevant =
      OkapiWrapper::compute_bf25f_derivative_wrt_summary_b(idf_collection,
          average_length_info, query_section, relevant_section,
          bm25f_parameter);

  return rnc_derivative_wrt_Y
      * (derivative_of_bm25f_wrt_summary_b_irrelevant
          - derivative_of_bm25f_wrt_summary_b_relevant);
}

double BM25FRankNetLearner::compute_rnc_derivative_wrt_description_weight(
    const double rnc_derivative_wrt_Y, IDFCollection* idf_collection,
    const OkapiWrapper::AverageLengthInfo& average_length_info,
    const StructuredSection& query_section,
    const StructuredSection& relevant_section,
    const StructuredSection& irrelevant_section,
    const BM25FParameter& bm25f_parameter) const {

  const double derivative_of_bm25f_wrt_description_weight_irrelevant =
      OkapiWrapper::compute_bf25f_derivative_wrt_description_weight(
          idf_collection, average_length_info, query_section,
          irrelevant_section, bm25f_parameter);

  const double derivative_of_bm25f_wrt_description_weight_relevant =
      OkapiWrapper::compute_bf25f_derivative_wrt_description_weight(
          idf_collection, average_length_info, query_section, relevant_section,
          bm25f_parameter);

  return rnc_derivative_wrt_Y
      * (derivative_of_bm25f_wrt_description_weight_irrelevant
          - derivative_of_bm25f_wrt_description_weight_relevant);
}

double BM25FRankNetLearner::compute_rnc_derivative_wrt_summary_weight(
    const double rnc_derivative_wrt_Y, IDFCollection* idf_collection,
    const OkapiWrapper::AverageLengthInfo& average_length_info,
    const StructuredSection& query_section,
    const StructuredSection& relevant_section,
    const StructuredSection& irrelevant_section,
    const BM25FParameter& bm25f_parameter) const {

  const double derivative_of_bm25f_wrt_summary_weight_irrelevant =
      OkapiWrapper::compute_bf25f_derivative_wrt_summary_weight(idf_collection,
          average_length_info, query_section, irrelevant_section,
          bm25f_parameter);

  const double derivative_of_bm25f_wrt_summary_weight_relevant =
      OkapiWrapper::compute_bf25f_derivative_wrt_summary_weight(idf_collection,
          average_length_info, query_section, relevant_section,
          bm25f_parameter);

  return rnc_derivative_wrt_Y
      * (derivative_of_bm25f_wrt_summary_weight_irrelevant
          - derivative_of_bm25f_wrt_summary_weight_relevant);
}

double BM25FRankNetLearner::compute_rnc_derivative_wrt_k1(
    const double rnc_derivative_wrt_Y, IDFCollection* idf_collection,
    const OkapiWrapper::AverageLengthInfo& average_length_info,
    const StructuredSection& query_section,
    const StructuredSection& relevant_section,
    const StructuredSection& irrelevant_section,
    const BM25FParameter& bm25f_parameter) const {

  const double derivative_of_bm25f_wrt_k_irrelevant =
      OkapiWrapper::compute_bm25f_derivative_wrt_k(idf_collection,
          average_length_info, query_section, irrelevant_section,
          bm25f_parameter);

  const double derivative_of_bm25f_wrt_k_relevant =
      OkapiWrapper::compute_bm25f_derivative_wrt_k(idf_collection,
          average_length_info, query_section, relevant_section,
          bm25f_parameter);

  return rnc_derivative_wrt_Y
      * (derivative_of_bm25f_wrt_k_irrelevant
          - derivative_of_bm25f_wrt_k_relevant);
}

void BM25FRankNetLearner::tune_parameters_on_one_pair(
    const RankNetTrainingInstance& training_pair, const double learning_rate) {
  /*
   * evaluate the gradient.
   *
   * 1) compute the RNC cost.
   */

  const ReportBuckets& report_buckets = this->get_report_buckets();
  IDFCollection* idf_collection = report_buckets.get_both_idf_collection();

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

  if (!(this->m_Unigram_bm25f_parameter.is_total_weight_fixed()
      && (this->m_Unigram_bm25f_parameter.get_total_weight() == 0))) {
    const StructuredSection& unigram_query_section =
        training_pair.get_query().get_Unigram_structured_section();
    const StructuredSection& unigram_relevant_section =
        training_pair.get_relevant_report().get_Unigram_structured_section();
    const StructuredSection& unigram_irrelevant_section =
        training_pair.get_irrelevant_report().get_Unigram_structured_section();

    const OkapiWrapper::AverageLengthInfo unigram_average_length_info(
        report_buckets.get_average_length_of_summary_Unigram_section(),
        report_buckets.get_average_length_of_description_Unigram_section());

    this->tune_gram_parameters_on_one_pair(learning_rate,
        derivative_of_rnc_wrt_Y, idf_collection, unigram_average_length_info,
        unigram_query_section, unigram_relevant_section,
        unigram_irrelevant_section, this->m_Unigram_bm25f_parameter,
        this->m_tuning_Unigram_bm25f_parameter);
  }

  if (!(this->m_Bigram_bm25f_parameter.is_total_weight_fixed()
      && (this->m_Bigram_bm25f_parameter.get_total_weight() == 0))) {

    const StructuredSection& bigram_query_section =
        training_pair.get_query().get_Bigram_structured_section();
    const StructuredSection& bigram_relevant_section =
        training_pair.get_relevant_report().get_Bigram_structured_section();
    const StructuredSection& bigram_irrelevant_section =
        training_pair.get_irrelevant_report().get_Bigram_structured_section();

    const OkapiWrapper::AverageLengthInfo bigram_average_length_info(
        report_buckets.get_average_length_of_summary_Bigram_section(),
        report_buckets.get_average_length_of_description_Bigram_section());

    this->tune_gram_parameters_on_one_pair(learning_rate,
        derivative_of_rnc_wrt_Y, idf_collection, bigram_average_length_info,
        bigram_query_section, bigram_relevant_section,
        bigram_irrelevant_section, this->m_Bigram_bm25f_parameter,
        this->m_tuning_Bigram_bm25f_parameter);
  }

}

float BM25FRankNetLearner::compute_term_weight_in_doc(const int summary_tf,
    const int desc_tf, const OkapiWrapper::LengthInfo& length_info,
    const OkapiWrapper::AverageLengthInfo& avg_length_info) const {
  return OkapiWrapper::compute_term_weight_for_bm25f(summary_tf, desc_tf,
      this->m_Unigram_bm25f_parameter, length_info, avg_length_info);
}

float BM25FRankNetLearner::compute_term_weight_in_query(const int summary_tf,
    const int desc_tf) const {
  return OkapiWrapper::compute_query_weight_for_bm25f(
      this->m_Unigram_bm25f_parameter, summary_tf, desc_tf);
}

float BM25FRankNetLearner::compute_similarity(
    const AbstractBugReport& query_report,
    const AbstractBugReport& base_report) const {
  const ReportBuckets& report_buckets = this->get_report_buckets();
  IDFCollection* idf_collection = report_buckets.get_both_idf_collection();

  double unigram_similarity = 0;
  if (!(this->m_Unigram_bm25f_parameter.is_total_weight_fixed()
      && (this->m_Unigram_bm25f_parameter.get_total_weight() == 0))) {
    const StructuredSection& unigram_query_section =
        query_report.get_Unigram_structured_section();
    const StructuredSection& unigram_base_section =
        base_report.get_Unigram_structured_section();

    const OkapiWrapper::AverageLengthInfo unigram_average_length_info(
        report_buckets.get_average_length_of_summary_Unigram_section(),
        report_buckets.get_average_length_of_description_Unigram_section());

    assert(
        this->m_Unigram_bm25f_parameter.get_total_weight() == this->m_Unigram_bm25f_parameter.get_total_weight());

    unigram_similarity = this->m_Unigram_bm25f_parameter.get_total_weight()
        * OkapiWrapper::compute_bm25f(idf_collection,
            unigram_average_length_info, unigram_query_section,
            unigram_base_section, this->m_Unigram_bm25f_parameter);
    assert(unigram_similarity == unigram_similarity);
  }

  double bigram_similarity = 0;
  if (!(this->m_Bigram_bm25f_parameter.is_total_weight_fixed()
      && (this->m_Bigram_bm25f_parameter.get_total_weight() == 0))) {
    const StructuredSection& bigram_query_section =
        query_report.get_Bigram_structured_section();
    const StructuredSection& bigram_base_section =
        base_report.get_Bigram_structured_section();

    const OkapiWrapper::AverageLengthInfo bigram_average_length_info(
        report_buckets.get_average_length_of_summary_Bigram_section(),
        report_buckets.get_average_length_of_description_Bigram_section());

    bigram_similarity = this->m_Bigram_bm25f_parameter.get_total_weight()
        * OkapiWrapper::compute_bm25f(idf_collection,
            bigram_average_length_info, bigram_query_section,
            bigram_base_section, this->m_Bigram_bm25f_parameter);
    assert(bigram_similarity == bigram_similarity);
  }
  const double result = unigram_similarity + bigram_similarity;
  assert(result == result);
  return result;
}

unsigned BM25FRankNetLearner::get_training_round_number() {
  if (this->m_default_model_parameter.is_k3_fixed()) {
    return 1;
  } else {
    return 2;
  }
}

void BM25FRankNetLearner::initialize_model_parameters(const unsigned round) {
  switch (round) {
  case 1: {
    this->m_Unigram_bm25f_parameter.initialize(
        this->m_default_model_parameter.get_unigram_weight(),
        this->m_default_model_parameter.is_unigram_weight_fixed(),
        this->m_default_model_parameter.get_k1(),
        this->m_default_model_parameter.is_k1_fixed(),
        this->m_default_model_parameter.get_summary_weight(),
        this->m_default_model_parameter.is_summary_weight_fixed(),
        this->m_default_model_parameter.get_summary_b(),
        this->m_default_model_parameter.is_summary_b_fixed(),
        this->m_default_model_parameter.get_description_weight(),
        this->m_default_model_parameter.is_description_weight_fixed(),
        this->m_default_model_parameter.get_description_b(),
        this->m_default_model_parameter.is_description_b_fixed(), 0, true);

    this->m_Bigram_bm25f_parameter.initialize(
        this->m_default_model_parameter.get_bigram_weight(),
        this->m_default_model_parameter.is_bigram_weight_fixed(),
        this->m_default_model_parameter.get_k1(),
        this->m_default_model_parameter.is_k1_fixed(),
        this->m_default_model_parameter.get_summary_weight(),
        this->m_default_model_parameter.is_summary_weight_fixed(),
        this->m_default_model_parameter.get_summary_b(),
        this->m_default_model_parameter.is_summary_b_fixed(),
        this->m_default_model_parameter.get_description_weight(),
        this->m_default_model_parameter.is_description_weight_fixed(),
        this->m_default_model_parameter.get_description_b(),
        this->m_default_model_parameter.is_description_b_fixed(), 0, true);
    break;
  }
  case 2: {
    if (2 > this->get_training_round_number()) {
      ERROR_HERE("illegal round.");
    }
    this->m_Unigram_bm25f_parameter.set_k3(0, false);
    this->m_Bigram_bm25f_parameter.set_k3(0, false);
    break;
  }
  default:
    ERROR_HERE("illegal round.");
    break;
  }
}

void BM25FRankNetLearner::learning_done() {
  this->m_Unigram_bm25f_parameter = this->m_best_Unigram_bm25f_parameter;
  this->m_Bigram_bm25f_parameter = this->m_best_Bigram_bm25f_parameter;
  this->m_Unigram_bm25f_parameter.print(this->get_log_file(), "unigram");
  this->m_Bigram_bm25f_parameter.print(this->get_log_file(), "bigram");
}

BM25FRankNetLearner::BM25FRankNetLearner(FILE* log_file,
    const ReportBuckets& buckets, const DefaultREPParameter& parameter) :
    AbstractRankNetLearner(log_file, buckets,
        parameter.get_count_of_irrelevant_reports_per_query(),
        parameter.get_max_query_count()), m_default_model_parameter(parameter) {
}

BM25FRankNetLearner::~BM25FRankNetLearner() {
}
