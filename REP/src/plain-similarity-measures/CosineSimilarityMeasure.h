/*
 * CosineSimilarityMeasure.h
 *
 *  Created on: Dec 15, 2010
 *      Author: Chengnian SUN
 */

#ifndef TF_BASEDCOSINESIMILARITYMEASURE_H_
#define TF_BASEDCOSINESIMILARITYMEASURE_H_

#include "IPlainSimilarityMeasure.h"
#include "../tfidf/IDFCollection.h"
#include <boost/unordered_map.hpp>

class CosineSimilarityMeasure: public IPlainSimilarityMeasure {
private:

  boost::unordered_map<int, boost::unordered_map<int, double>*> m_report_term_weights_map;

  int m_summary_weight;

  boost::unordered_map<int, double>* get_term_weight_vector(
      const AbstractBugReport& report, const ReportBuckets& bucket);

  void compute_term_weights(const AbstractBugReport& report,
      boost::unordered_map<int, double>& result_collector,
      const ReportBuckets& bucket);

  virtual double weigh_term(const int tf, const double idf) = 0;

public:

  virtual double compute_similarity(const AbstractBugReport& query_report,
      const AbstractBugReport& doc_report, const ReportBuckets& bucket);

  CosineSimilarityMeasure(const int summary_weight) :
      m_summary_weight(summary_weight) {
  }

  virtual ~CosineSimilarityMeasure();
};

#endif /* TF_BasedCosineSimilarityMeasure_H_ */
