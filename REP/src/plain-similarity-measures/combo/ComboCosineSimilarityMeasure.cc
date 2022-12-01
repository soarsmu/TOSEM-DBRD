/*
 * ComboCosineSimilarityMeasure.cc
 *
 *  Created on: Dec 15, 2010
 *      Author: Chengnian SUN
 */

#include "ComboCosineSimilarityMeasure.h"
#include "../../tfidf/IDFCollection.h"

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

void ComboCosineSimilarityMeasure::compute_description_term_weights(
    const AbstractBugReport& report,
    boost::unordered_map<int, double>& result_collector,
    const ReportBuckets& bucket) {
  assert(result_collector.empty());

  temp_term_frequency_cache.clear();

  //	update_tfs_with_section(temp_term_frequency_cache, report.get_summary_unigrams().get_terms(), 1);
  update_tfs_with_section(temp_term_frequency_cache,
      report.get_description_unigrams().get_terms(), 1);

  double length = 0;
  for (boost::unordered_map<int, int>::const_iterator iter =
      temp_term_frequency_cache.cbegin();
      iter != temp_term_frequency_cache.cend(); iter++) {
    const int term_id = iter->first;
    double weight = this->weigh_term(iter->second,
        bucket.get_idf_collection(IDFCollectionType::IDF_DESC)->get_idf(
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

void ComboCosineSimilarityMeasure::compute_summary_term_weights(
    const AbstractBugReport& report,
    boost::unordered_map<int, double>& result_collector,
    const ReportBuckets& bucket) {
  assert(result_collector.empty());

  temp_term_frequency_cache.clear();

  update_tfs_with_section(temp_term_frequency_cache,
      report.get_summary_unigrams().get_terms(), 1);
  //	update_tfs_with_section(temp_term_frequency_cache, report.get_description_unigrams().get_terms(), 1);

  double length = 0;
  for (boost::unordered_map<int, int>::const_iterator iter =
      temp_term_frequency_cache.cbegin();
      iter != temp_term_frequency_cache.cend(); iter++) {
    const int term_id = iter->first;
    double weight = this->weigh_term(iter->second,
        bucket.get_idf_collection(IDFCollectionType::IDF_SUMM)->get_idf(
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

boost::unordered_map<int, double>* ComboCosineSimilarityMeasure::get_summary_term_weight_vector(
    const AbstractBugReport& report, const ReportBuckets& bucket) {

  boost::unordered_map<int, double>* term_weight_vector;

  const int report_id = report.get_id();
  if (this->m_summary_term_weights_map.find(report_id)
      == this->m_summary_term_weights_map.end()) {
    term_weight_vector = new boost::unordered_map<int, double>();
    this->m_summary_term_weights_map[report_id] = term_weight_vector;
    this->compute_summary_term_weights(report, *term_weight_vector, bucket);
  } else {
    term_weight_vector = this->m_summary_term_weights_map[report_id];
  }

  return term_weight_vector;
}

boost::unordered_map<int, double>* ComboCosineSimilarityMeasure::get_description_term_weight_vector(
    const AbstractBugReport& report, const ReportBuckets& bucket) {

  boost::unordered_map<int, double>* term_weight_vector;

  const int report_id = report.get_id();
  if (this->m_description_term_weights_map.find(report_id)
      == this->m_description_term_weights_map.end()) {
    term_weight_vector = new boost::unordered_map<int, double>();
    this->m_description_term_weights_map[report_id] = term_weight_vector;
    this->compute_description_term_weights(report, *term_weight_vector, bucket);
  } else {
    term_weight_vector = this->m_description_term_weights_map[report_id];
  }

  return term_weight_vector;
}

double ComboCosineSimilarityMeasure::compute_summary_similarity(
    const AbstractBugReport& query_report, const AbstractBugReport& doc_report,
    const ReportBuckets& bucket) {
  boost::unordered_map<int, double>* doc_term_weights =
      this->get_summary_term_weight_vector(doc_report, bucket);

  boost::unordered_map<int, double>* query_term_weights =
      this->get_summary_term_weight_vector(query_report, bucket);

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

double ComboCosineSimilarityMeasure::compute_description_similarity(
    const AbstractBugReport& query_report, const AbstractBugReport& doc_report,
    const ReportBuckets& bucket) {
  boost::unordered_map<int, double>* doc_term_weights =
      this->get_description_term_weight_vector(doc_report, bucket);

  boost::unordered_map<int, double>* query_term_weights =
      this->get_description_term_weight_vector(query_report, bucket);

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

double ComboCosineSimilarityMeasure::compute_similarity(
    const AbstractBugReport& query_report, const AbstractBugReport& doc_report,
    const ReportBuckets& bucket) {
  const double summary_similarity = this->compute_summary_similarity(
      query_report, doc_report, bucket);
  const double description_similarity = this->compute_description_similarity(
      query_report, doc_report, bucket);
  return this->m_summary_weight * summary_similarity + description_similarity;
}
