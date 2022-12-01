/*
 * ComboCosineSimilarityMeasure.h
 *
 *  Created on: Dec 15, 2010
 *      Author: Chengnian SUN
 */

#ifndef COMBOCOSINESIMILARITYMEASURE_H_
#define COMBOCOSINESIMILARITYMEASURE_H_

#include "../IPlainSimilarityMeasure.h"
#include <boost/unordered_map.hpp>

class ComboCosineSimilarityMeasure: public IPlainSimilarityMeasure {
private:

  int m_summary_weight;

  boost::unordered_map<int, boost::unordered_map<int, double>*> m_summary_term_weights_map;

  boost::unordered_map<int, boost::unordered_map<int, double>*> m_description_term_weights_map;

  virtual double weigh_term(const int tf, const double idf) = 0;

  double compute_summary_similarity(const AbstractBugReport& query_report,
      const AbstractBugReport& doc_report, const ReportBuckets& bucket);

  double compute_description_similarity(const AbstractBugReport& query_report,
      const AbstractBugReport& doc_report, const ReportBuckets& bucket);

  boost::unordered_map<int, double>* get_summary_term_weight_vector(
      const AbstractBugReport& report, const ReportBuckets& bucket);

  boost::unordered_map<int, double>* get_description_term_weight_vector(
      const AbstractBugReport& report, const ReportBuckets& bucket);

  void compute_summary_term_weights(const AbstractBugReport& report,
      boost::unordered_map<int, double>& result_collector,
      const ReportBuckets& bucket);

  void compute_description_term_weights(const AbstractBugReport& report,
      boost::unordered_map<int, double>& result_collector,
      const ReportBuckets& bucket);

public:

  virtual double compute_similarity(const AbstractBugReport& query_report,
      const AbstractBugReport& doc_report, const ReportBuckets& bucket);

  ComboCosineSimilarityMeasure(const int summary_weight) :
      m_summary_weight(summary_weight) {
  }

  virtual ~ComboCosineSimilarityMeasure() {

  }
};

#endif /* COMBOCOSINESIMILARITYMEASURE_H_ */
