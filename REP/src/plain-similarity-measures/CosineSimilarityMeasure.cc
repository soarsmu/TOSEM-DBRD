/*
 * CosineSimilarityMeasure.cc
 *
 *  Created on: Dec 15, 2010
 *      Author: Chengnian SUN
 */

#include "CosineSimilarityMeasure.h"

static boost::unordered_map<int, int> temp_term_frequency_cache;

static inline void update_tfs_with_section(
    boost::unordered_map<int, int>& weights, const vector<Term>& section_terms,
    const int term_weight) {
  const unsigned size = section_terms.size();
  for (unsigned i = 0; i < size; i++) {
    const Term& term = section_terms[i];
    weights[term.get_tid()] += term_weight * term.get_term_frequency();
  }
}

void CosineSimilarityMeasure::compute_term_weights(
    const AbstractBugReport& report,
    boost::unordered_map<int, double>& result_collector,
    const ReportBuckets& bucket) {
  assert(result_collector.empty());

  temp_term_frequency_cache.clear();

  update_tfs_with_section(temp_term_frequency_cache,
      report.get_summary_unigrams().get_terms(), this->m_summary_weight);
  update_tfs_with_section(temp_term_frequency_cache,
      report.get_description_unigrams().get_terms(), 1);

  double length = 0;
  for (boost::unordered_map<int, int>::const_iterator iter =
      temp_term_frequency_cache.cbegin();
      iter != temp_term_frequency_cache.cend(); iter++) {
    const int term_id = iter->first;
    double weight = this->weigh_term(iter->second,
        bucket.get_idf_collection(IDFCollectionType::IDF_BOTH)->get_idf(
            term_id));
    result_collector[term_id] = weight;
    length += weight * weight;
  }

  length = sqrt(length);
  assert(length == length);
  for (boost::unordered_map<int, double>::iterator iter =
      result_collector.begin(); iter != result_collector.end(); iter++) {
    iter->second /= length;
  }
}

boost::unordered_map<int, double>* CosineSimilarityMeasure::get_term_weight_vector(
    const AbstractBugReport& report, const ReportBuckets& bucket) {

  boost::unordered_map<int, double>* term_weight_vector;

  const int report_id = report.get_id();
  if (this->m_report_term_weights_map.find(report_id)
      == this->m_report_term_weights_map.end()) {
    term_weight_vector = new boost::unordered_map<int, double>();
    this->m_report_term_weights_map[report_id] = term_weight_vector;
    this->compute_term_weights(report, *term_weight_vector, bucket);
  } else {
    term_weight_vector = this->m_report_term_weights_map[report_id];
  }

  return term_weight_vector;
}

double CosineSimilarityMeasure::compute_similarity(
    const AbstractBugReport& query_report, const AbstractBugReport& doc_report,
    const ReportBuckets& bucket) {
  boost::unordered_map<int, double>* doc_term_weights =
      this->get_term_weight_vector(doc_report, bucket);

  boost::unordered_map<int, double>* query_term_weights =
      this->get_term_weight_vector(query_report, bucket);

  double result = 0;
  for (boost::unordered_map<int, double>::iterator iter =
      query_term_weights->begin(); iter != query_term_weights->end(); iter++) {
    if (doc_term_weights->find(iter->first) == doc_term_weights->end()) {
      continue;
    } else {
      result += iter->second * (*doc_term_weights)[iter->first];
    }
  }
  return result;
}

CosineSimilarityMeasure::~CosineSimilarityMeasure() {
  for (boost::unordered_map<int, boost::unordered_map<int, double>*>::iterator iter =
      this->m_report_term_weights_map.begin();
      iter != this->m_report_term_weights_map.end(); iter++) {
    if (iter->second != NULL) {
      delete iter->second;
      iter->second = NULL;
    }
  }
}
