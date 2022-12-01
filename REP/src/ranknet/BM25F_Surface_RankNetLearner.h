/*
 * BM25F_Surface_RankNetLearner.h
 *
 *  Created on: Jan 10, 2011
 *      Author: Chengnian SUN
 */

#ifndef BM25F_SURFACE_RANKNETLEARNER_H_
#define BM25F_SURFACE_RANKNETLEARNER_H_

#include "BM25FRankNetLearner.h"
#include "SurfaceWeight.h"

class BM25F_Surface_RankNetLearner: public BM25FRankNetLearner {
private:

  SurfaceWeight m_surface_weight;

  SurfaceWeight m_tuning_surface_weight;

  SurfaceWeight m_best_surface_weight;

protected:

  //--------------------------------------------------------------
  // compute derivatives
  //--------------------------------------------------------------

  typedef float (*SurfaceFunctionPointer)(const AbstractBugReport& query_report,
      const AbstractBugReport& base_report);

  double compute_rnc_derivative_wrt_non_textual_weight(
      const double rnc_derivative_wrt_Y, const AbstractBugReport& query_report,
      const AbstractBugReport& relevant_report,
      const AbstractBugReport& irrelevant_report,
      SurfaceFunctionPointer surface_function_pointer) const {
    const double irrelevant_derivative = surface_function_pointer(query_report,
        irrelevant_report);
    const double relevant_derivative = surface_function_pointer(query_report,
        relevant_report);
    return rnc_derivative_wrt_Y * (irrelevant_derivative - relevant_derivative);
  }

  //--------------------------------------------------------------
  // END compute derivatives
  //--------------------------------------------------------------

  virtual void before_tune_parameters_on_one_pair();

  virtual void after_tune_parameters_on_one_pair();

  virtual void tune_parameters_on_one_pair(
      const RankNetTrainingInstance& training_pair, const double learning_rate);

  virtual void initialize_model_parameters(const unsigned round);

  virtual void learning_done();

  virtual void find_a_better_model();

  float compute_categorial_similarity(const AbstractBugReport& query,
      const AbstractBugReport& base) const;

public:

  virtual float compute_similarity_with_textual_sim(
      const float textual_similarity, const AbstractBugReport& query,
      const AbstractBugReport& base) const;

  virtual float compute_similarity(const AbstractBugReport& query_report,
      const AbstractBugReport& base_report) const;

  BM25F_Surface_RankNetLearner(FILE* log_file,
      const ReportBuckets& report_buckets,
      const DefaultREPParameter& parameter);

  virtual ~BM25F_Surface_RankNetLearner();
};

#endif /* BM25F_SURFACE_RANKNETLEARNER_H_ */
