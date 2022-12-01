/*
 * RankNetLearner.h
 *
 *  Created on: Jan 4, 2011
 *      Author: Chengnian SUN
 */

#ifndef _BM25F_RANK_NET_LEARNER_H_
#define _BM25F_RANK_NET_LEARNER_H_

#include <algorithm>
#include <cassert>
#include <cmath>

#include "../okapi/BM25FParameter.h"
#include "../okapi/OkapiWrapper.h"
#include "AbstractRankNetLearner.h"

using namespace std;

class IDFCollection;
class DefaultREPParameter;

class BM25FRankNetLearner: public AbstractRankNetLearner {
private:

  BM25FParameter m_Unigram_bm25f_parameter;

  BM25FParameter m_tuning_Unigram_bm25f_parameter;

  BM25FParameter m_Bigram_bm25f_parameter;

  BM25FParameter m_tuning_Bigram_bm25f_parameter;

  BM25FParameter m_best_Unigram_bm25f_parameter;

  BM25FParameter m_best_Bigram_bm25f_parameter;

  void tune_gram_parameters_on_one_pair(const double learning_weight,
      const double derivative_of_rnc_wrt_Y, IDFCollection* idf_collection,
      OkapiWrapper::AverageLengthInfo average_length_info,
      const StructuredSection& query_section,
      const StructuredSection& relevant_section,
      const StructuredSection& irrelevant_section,
      const BM25FParameter& currentBm25fparameter,
      BM25FParameter& parameterToTune);

protected:

  const DefaultREPParameter& m_default_model_parameter;

  double
  compute_rnc_derivative_wrt_textual_weight(const double rnc_derivative_wrt_Y,
      IDFCollection* idf_collection,
      const OkapiWrapper::AverageLengthInfo& average_length_info,
      const StructuredSection& query_section,
      const StructuredSection& relevant_section,
      const StructuredSection& irrelevant_section,
      const BM25FParameter& bm25f_parameter) const;

  double
  compute_rnc_derivative_wrt_k1(const double rnc_derivative_wrt_Y,
      IDFCollection* idf_collection,
      const OkapiWrapper::AverageLengthInfo& average_length_info,
      const StructuredSection& query_section,
      const StructuredSection& relevant_section,
      const StructuredSection& irrelevant_section,
      const BM25FParameter& bm25f_parameter) const;

  double
  compute_rnc_derivative_wrt_summary_weight(const double rnc_derivative_wrt_Y,
      IDFCollection* idf_collection,
      const OkapiWrapper::AverageLengthInfo& average_length_info,
      const StructuredSection& query_section,
      const StructuredSection& relevant_section,
      const StructuredSection& irrelevant_section,
      const BM25FParameter& bm25f_parameter) const;

  double
  compute_rnc_derivative_wrt_summary_b(const double rnc_derivative_wrt_Y,
      IDFCollection* idf_collection,
      const OkapiWrapper::AverageLengthInfo& average_length_info,
      const StructuredSection& query_section,
      const StructuredSection& relevant_section,
      const StructuredSection& irrelevant_section,
      const BM25FParameter& bm25f_parameter) const;

  double
  compute_rnc_derivative_wrt_description_b(const double rnc_derivative_wrt_Y,
      IDFCollection* idf_collection,
      const OkapiWrapper::AverageLengthInfo& average_length_info,
      const StructuredSection& query_section,
      const StructuredSection& relevant_section,
      const StructuredSection& irrelevant_section,
      const BM25FParameter& bm25f_parameter) const;

  double
  compute_rnc_derivative_wrt_description_weight(
      const double rnc_derivative_wrt_Y, IDFCollection* idf_collection,
      const OkapiWrapper::AverageLengthInfo& average_length_info,
      const StructuredSection& query_section,
      const StructuredSection& relevant_section,
      const StructuredSection& irrelevant_section,
      const BM25FParameter& bm25f_parameter) const;

  double
  compute_rnc_derivative_wrt_k3(const double rnc_derivative_wrt_Y,
      IDFCollection* idf_collection,
      const OkapiWrapper::AverageLengthInfo& average_length_info,
      const StructuredSection& query_section,
      const StructuredSection& relevant_section,
      const StructuredSection& irrelevant_section,
      const BM25FParameter& bm25f_parameter) const;

  virtual void before_tune_parameters_on_one_pair();

  virtual void after_tune_parameters_on_one_pair();

  virtual void tune_parameters_on_one_pair(
      const RankNetTrainingInstance& training_pair, const double learning_rate);

  virtual void learning_done();

  virtual unsigned get_training_round_number();

  virtual void initialize_model_parameters(const unsigned round);

  virtual void find_a_better_model();

public:

  BM25FRankNetLearner(FILE* log_file, const ReportBuckets& report_buckets,
      const DefaultREPParameter& parameter);

  virtual float compute_similarity(const AbstractBugReport& query_report,
      const AbstractBugReport& base_report) const;

  virtual float compute_term_weight_in_doc(const int summary_tf,
      const int desc_tf, const OkapiWrapper::LengthInfo& length_info,
      const OkapiWrapper::AverageLengthInfo& avg_length_info) const;

  virtual float compute_term_weight_in_query(const int summary_tf,
      const int desc_tf) const;

  virtual float compute_similarity_with_textual_sim(
      const float textual_similarity, const AbstractBugReport& query,
      const AbstractBugReport& base) const;

  virtual float weigh_textual_similarity(const float textual_similarity) const;

  virtual ~BM25FRankNetLearner();

  virtual void learn();

};

#endif /* _BM25F_RANK_NET_LEARNER_H_ */
