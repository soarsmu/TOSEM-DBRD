/*
 * CrossFieldsFeatureValueCalculator.cc
 *
 *  Created on: Jan 6, 2011
 *      Author: Chengnian SUN
 */

#include "CrossFieldsFeatureValueCalculator.h"

#include "../detection-model/ReportBuckets.h"
#include "../report-model/DuplicateBugReport.h"
#include "../report-model/MasterBugReport.h"
#include "../tfidf/IDFCollection.h"

double CrossFieldsFeatureValueCalculator::compute_feature_value(
    const AbstractBugReport& query, const AbstractBugReport& report) {
  double similarity = 0;
  unsigned query_index = 0;
  unsigned doc_index = 0;

  const ReportBuckets& report_buckets = this->get_report_buckets();
  const Section& query_section = query.get_section(this->m_query_type);
  const Section& doc_section = report.get_section(this->m_doc_type);
  IDFCollection* idf_collection = report_buckets.get_idf_collection(
      this->m_idf_type);

  const vector<Term>& query_terms = query_section.get_terms();
  const vector<Term>& doc_terms = doc_section.get_terms();

  const unsigned count_of_query_terms = query_terms.size();
  const unsigned count_of_doc_terms = doc_terms.size();

  const unsigned doc_length = doc_section.get_length();
  const double average_length_of_doc_section =
      report_buckets.get_average_length_of_sections(this->m_doc_type);

  while (query_index < count_of_query_terms && doc_index < count_of_doc_terms) {
    const Term& query_term = query_terms[query_index];
    const int query_term_id = query_term.get_tid();

    const Term& doc_term = doc_terms[doc_index];
    const int doc_term_id = doc_term.get_tid();
    if (query_term_id < doc_term_id) {
      query_index++;
    } else if (query_term_id > doc_term_id) {
      doc_index++;
    } else {
      const double idf = idf_collection->get_idf(query_term_id);
      const double query_tf = query_term.get_term_frequency();
      const double doc_tf = doc_term.get_term_frequency();
      similarity += this->weigh_term(idf, doc_tf, query_tf, doc_length,
          average_length_of_doc_section);
      //			similarity += this->get_weight(feature_index, idf_collection, query_term, doc_term, doc_length,
      //					average_length_of_doc_section);
      query_index++;
      doc_index++;
    }
  }
  return similarity;
}

CrossFieldsFeatureValueCalculator::CrossFieldsFeatureValueCalculator(
    const ReportBuckets& report_buckets,
    const SectionType::EnumSectionType query_type,
    const SectionType::EnumSectionType doc_type,
    const IDFCollectionType::EnumIDFCollectionType idf_type) :
    AbstractFeatureValueCalculator(report_buckets), m_query_type(query_type), m_doc_type(
        doc_type), m_idf_type(idf_type) {
}

CrossFieldsFeatureValueCalculator::~CrossFieldsFeatureValueCalculator() {
}
