/*
 * AbstractRankNetLearner.h
 *
 *  Created on: Jan 10, 2011
 *      Author: Chengnian SUN
 */

#ifndef ABSTRACTRANKNETLEARNER_H_
#define ABSTRACTRANKNETLEARNER_H_

#include <cassert>
#include <climits>
#include <cstdio>
#include <vector>

#include "../okapi/OkapiWrapper.h"
#include "RankNetTrainingInstance.h"

using namespace std;

class AbstractBugReport;
class MasterBugReport;
class DuplicateBugReport;
class DefaultREPParameter;
class ReportBuckets;

class AbstractRankNetLearner {
private:

  const ReportBuckets& m_buckets;

  FILE* m_log_file;

  vector<RankNetTrainingInstance> m_training_pairs;

  vector<RankNetTrainingInstance> m_validating_pairs;

  //	double m_best_map_so_far;
  double m_best_rnc_so_far;

  const unsigned m_count_of_irrelevant_reports_per_query;

  const unsigned m_max_query_count;

  const string m_training_file;

private:

  void build_training_pairs_for_a_pair_of_duplicates(const int bucket_id,
      const vector<MasterBugReport*>& masters,
      const AbstractBugReport* relevant_1, const AbstractBugReport* relevant_2);

  vector<MasterBugReport*> get_random_master_reports(
      const vector<MasterBugReport*>& masters, unsigned count_to_random,
      int excluding_master_id);

  double compute_total_rnc_cost(
      const vector<RankNetTrainingInstance>& training_pairs) const;

  void build_training_pairs();

  void build_training_pairs_from_file(string training_file);

  const static unsigned MAX_EPOCHES = 24;

  constexpr static double INITIAL_LEARNING_RATE = 0.001;

  double compute_bucket_similarity(DuplicateBugReport& query_report,
      MasterBugReport& bucket_master) const;

protected:

  FILE* get_log_file();

  const ReportBuckets& get_report_buckets() const;

  /*
   * RNC(Y) = log(1 + exp(Y)) where Y = y2 - y1, y2 and y1 are the similarity scores.
   *
   * RNC'(Y) = exp(Y) / (1 + exp(Y))
   *
   */
  double compute_rnc_derivative_wrt_Y(
      const RankNetTrainingInstance& training_pair) const;

  double compute_similarity_difference(
      const RankNetTrainingInstance& training_pair) const;

  /**
   * Must be implemented.
   */
  virtual void initialize_model_parameters(const unsigned round) = 0;

  virtual void before_tune_parameters_on_one_pair() = 0;

  virtual void after_tune_parameters_on_one_pair() = 0;

  /**
   * Must be implemented.
   */
  virtual void tune_parameters_on_one_pair(
      const RankNetTrainingInstance& training_pair,
      const double learning_rate) = 0;

  /**
   * Must be implemented.
   */
  virtual void learning_done() = 0;

  virtual unsigned get_training_round_number() = 0;

  virtual void find_a_better_model() = 0;

  AbstractRankNetLearner(FILE* log_file, const ReportBuckets& report_buckets,
      unsigned count_of_irrelevant_reports_per_query, unsigned max_query_count);

public:

  void learn();

  virtual float compute_similarity(const AbstractBugReport& query_report,
      const AbstractBugReport& base_report) const = 0;

  virtual float compute_term_weight_in_doc(const int summary_tf,
      const int desc_tf, const OkapiWrapper::LengthInfo& length_info,
      const OkapiWrapper::AverageLengthInfo& avg_length_info) const = 0;

  virtual float compute_term_weight_in_query(const int summary_tf,
      const int desc_tf) const = 0;

  /**
   * this method computes the similarity between two reports.
   * But, the parameter "textual_similarity" indicates the similarity,
   * so the method should compute the report similarity based on the
   * given textual similarity.
   *
   * This method is mainly used for the retrieval using index, where
   * the textual similarity is computed during retrieval, the method
   * can complete the similarity computation by incorporating the
   * categorical similarity, and the weight for the textual similarity.
   */
  virtual float compute_similarity_with_textual_sim(
      const float textual_similarity, const AbstractBugReport& query,
      const AbstractBugReport& base) const = 0;

  /**
   * weigh the given textual similarity.
   */
  virtual float weigh_textual_similarity(
      const float textual_similarity) const =0;

  virtual ~AbstractRankNetLearner();
};

///////////////////////////////////////////////////////////////////////////////
inline FILE* AbstractRankNetLearner::get_log_file() {
  return this->m_log_file;
}

inline const ReportBuckets& AbstractRankNetLearner::get_report_buckets() const {
  return this->m_buckets;
}

#endif /* ABSTRACTRANKNETLEARNER_H_ */
