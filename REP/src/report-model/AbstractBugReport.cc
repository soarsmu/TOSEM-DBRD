/*
 * BugReport.cc
 *
 *  Created on: Jan 6, 2011
 *      Author: Chengnian SUN
 */

#include "AbstractBugReport.h"

AbstractBugReport::~AbstractBugReport() {
  delete this->m_similarity_info;
}

const Section& AbstractBugReport::get_section(
    enum SectionType::EnumSectionType type) const {
  switch (type) {
  case SectionType::SUM_UNI:
    return this->m_summary_unigrams;
  case SectionType::SUM_BI:
    return this->m_summary_bigrams;
  case SectionType::SUM_TRI:
    return this->m_summary_trigrams;

  case SectionType::DESC_UNI:
    return this->m_description_unigrams;
  case SectionType::DESC_BI:
    return this->m_description_bigrams;
  case SectionType::DESC_TRI:
    return this->m_description_trigrams;

  case SectionType::ALL_UNI:
    return this->m_all_unigrams;
  case SectionType::ALL_BI:
    return this->m_all_bigrams;
  case SectionType::ALL_TRI:
    return this->m_all_trigrams;
  default:
    assert(false);
    return this->m_summary_unigrams;
  }
}

vector<StructuredTerm> StructuredSection::compute_terms(
    const Section& summary_section, const Section& description_section) {
  vector<StructuredTerm> terms;
  const vector<Term>& summary_terms = summary_section.get_terms();
  const vector<Term>& desc_terms = description_section.get_terms();
  const unsigned count_of_summary_terms = summary_terms.size();
  const unsigned count_of_desc_terms = desc_terms.size();

  unsigned summary_index = 0;
  unsigned desc_index = 0;
  while (summary_index < count_of_summary_terms
      && desc_index < count_of_desc_terms) {
    const Term& summary_term = summary_terms[summary_index];
    const int summary_tid = summary_term.get_tid();
    const Term& desc_term = desc_terms[desc_index];
    const int desc_tid = desc_term.get_tid();
    if (summary_tid > desc_tid) {
      terms.push_back(
          StructuredTerm(desc_tid, 0, desc_term.get_term_frequency()));
      desc_index++;
    } else if (summary_tid < desc_tid) {
      terms.push_back(
          StructuredTerm(summary_tid, summary_term.get_term_frequency(), 0));
      summary_index++;
    } else {
      // summary_term.get_tid() == desc_term.get_tid()
      terms.push_back(
          StructuredTerm(summary_tid, summary_term.get_term_frequency(),
              desc_term.get_term_frequency()));
      desc_index++;
      summary_index++;
    }
  }
  while (summary_index < count_of_summary_terms) {
    const Term& summary_term = summary_terms[summary_index];
    terms.push_back(
        StructuredTerm(summary_term.get_tid(),
            summary_term.get_term_frequency(), 0));
    summary_index++;
  }
  while (desc_index < count_of_desc_terms) {
    const Term& desc_term = desc_terms[desc_index];
    terms.push_back(
        StructuredTerm(desc_term.get_tid(), 0, desc_term.get_term_frequency()));
    desc_index++;
  }
  return terms;
}

AbstractBugReport::AbstractBugReport(const AbstractBugReport& other) :
    m_id(other.m_id), m_duplicate_id(other.m_duplicate_id), m_similarity_info(
        new SimilarityInfo()), m_summary_unigrams(other.m_summary_unigrams), m_summary_bigrams(
        other.m_summary_bigrams), m_summary_trigrams(other.m_summary_trigrams), m_description_unigrams(
        other.m_description_unigrams), m_description_bigrams(
        other.m_description_bigrams), m_description_trigrams(
        other.m_description_trigrams), m_all_unigrams(other.m_all_unigrams), m_all_bigrams(
        other.m_all_bigrams), m_all_trigrams(other.m_all_trigrams), m_structured_unigrams(
        other.m_structured_unigrams), m_structured_bigrams(
        other.m_structured_bigrams), m_version(other.m_version), m_component(
        other.m_component), m_sub_component(other.m_sub_component), m_report_type(
        other.m_report_type), m_priority(other.m_priority), m_timestamp_in_days(
        other.m_timestamp_in_days) {

}

AbstractBugReport::AbstractBugReport(const int id, const int duplicate_id,
    const vector<Term>& summary_unigrams, const vector<Term>& summary_bigrams,
    const vector<Term>& summary_trigrams,
    const vector<Term>& descrption_unigrams,
    const vector<Term>& description_bigrams,
    const vector<Term>& description_trigrams, const vector<Term>& all_unigrams,
    const vector<Term>& all_bigrams, const vector<Term>& all_trigrams,
    const int version, const int component, const int sub_component,
    const int report_type, const int priority, const unsigned timestamp_in_days) :
    m_id(id), m_duplicate_id(duplicate_id), m_similarity_info(
        new SimilarityInfo()), m_summary_unigrams(summary_unigrams), m_summary_bigrams(
        summary_bigrams), m_summary_trigrams(summary_trigrams), m_description_unigrams(
        descrption_unigrams), m_description_bigrams(description_bigrams), m_description_trigrams(
        description_trigrams), m_all_unigrams(all_unigrams), m_all_bigrams(
        all_bigrams), m_all_trigrams(all_trigrams), m_structured_unigrams(
        summary_unigrams, descrption_unigrams), m_structured_bigrams(
        summary_bigrams, description_bigrams), m_version(version), m_component(
        component), m_sub_component(sub_component), m_report_type(report_type), m_priority(
        priority), m_timestamp_in_days(timestamp_in_days) {
}
