/*
 * BugReport.h
 *
 *  Created on: 2010-7-29
 *      Author: Chengnian Sun
 */

#ifndef _BUG_REPORT_H_
#define _BUG_REPORT_H_

#include <boost/unordered_map.hpp>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "Section.h"
#include "SectionType.h"
#include "SimilarityInfo.h"

using namespace std;
class MasterBugReport;

class AbstractBugReport {
private:

  // the identity of the current bug report
  const int m_id;

  // the identity of the master report if the current report is a duplicate.
  // if duplicateId < 0 || duplicateId == id, then the current report is not a duplicate.
  const int m_duplicate_id;

  SimilarityInfo* m_similarity_info;

  const Section m_summary_unigrams;

  const Section m_summary_bigrams;

  const Section m_summary_trigrams;

  const Section m_description_unigrams;

  const Section m_description_bigrams;

  const Section m_description_trigrams;

  const Section m_all_unigrams;

  const Section m_all_bigrams;

  const Section m_all_trigrams;

  // ----------------------------

  StructuredSection m_structured_unigrams;

  StructuredSection m_structured_bigrams;

  // ----------------------------

  const int m_version;

  const int m_component;

  const int m_sub_component;

  const int m_report_type;

  const int m_priority;

  const unsigned m_timestamp_in_days;

public:

  virtual const MasterBugReport* get_master() const = 0;

  AbstractBugReport(const AbstractBugReport& other);

  virtual AbstractBugReport* get_copy() const = 0;

  virtual void set_detected() = 0;

  virtual bool is_detected() const = 0;

  const StructuredSection& get_Bigram_structured_section() const;

  const StructuredSection& get_Unigram_structured_section() const;

  int get_version() const;

  int get_component() const;

  int get_sub_component() const;

  int get_report_type() const;

  int get_priority() const;

  unsigned get_timestamp_in_days() const;

  SimilarityInfo* get_similarity_info() const;

  bool is_duplicate() const;

  int get_id() const;

  int get_duplicate_id() const;

  const Section& get_section(enum SectionType::EnumSectionType type) const;

  const Section& get_summary_unigrams() const;

  inline const Section& get_summary_bigrams() const;

  inline const Section& get_summary_trigrams() const;

  inline const Section& get_description_unigrams() const;

  inline const Section& get_description_bigrams() const;

  inline const Section& get_description_trigrams() const;

  inline const Section& get_all_unigrams() const;

  inline const Section& get_all_bigrams() const;

  inline const Section& get_all_trigrams() const;

  AbstractBugReport(const int id, const int duplicate_id,
      const vector<Term>& summary_unigrams, const vector<Term>& summary_bigrams,
      const vector<Term>& summary_trigrams,
      const vector<Term>& descrption_unigrams,
      const vector<Term>& description_bigrams,
      const vector<Term>& description_trigrams,
      const vector<Term>& all_unigrams, const vector<Term>& all_bigrams,
      const vector<Term>& all_trigrams, const int version, const int component,
      const int sub_component, const int report_type, const int priority,
      const unsigned timestamp_in_days);

  virtual ~AbstractBugReport();

};

inline const StructuredSection& AbstractBugReport::get_Bigram_structured_section() const {
  return this->m_structured_bigrams;
}

inline const StructuredSection& AbstractBugReport::get_Unigram_structured_section() const {
  return this->m_structured_unigrams;
}

inline int AbstractBugReport::get_version() const {
  return this->m_version;
}

inline int AbstractBugReport::get_component() const {
  return this->m_component;
}

inline int AbstractBugReport::get_sub_component() const {
  return this->m_sub_component;
}

inline int AbstractBugReport::get_report_type() const {
  return this->m_report_type;
}

inline int AbstractBugReport::get_priority() const {
  return this->m_priority;
}

inline SimilarityInfo* AbstractBugReport::get_similarity_info() const {
  assert(this->m_similarity_info);
  return this->m_similarity_info;
}

inline bool AbstractBugReport::is_duplicate() const {
  assert(this->m_duplicate_id > 0);
  return this->m_duplicate_id != this->m_id;
}

inline int AbstractBugReport::get_id() const {
  return this->m_id;
}

inline int AbstractBugReport::get_duplicate_id() const {
  return this->m_duplicate_id;
}

inline const Section& AbstractBugReport::get_summary_unigrams() const {
  return this->m_summary_unigrams;
}

inline const Section& AbstractBugReport::get_summary_bigrams() const {
  return this->m_summary_bigrams;
}

inline const Section& AbstractBugReport::get_summary_trigrams() const {
  return this->m_summary_trigrams;
}

inline const Section& AbstractBugReport::get_description_unigrams() const {
  return this->m_description_unigrams;
}

inline const Section& AbstractBugReport::get_description_bigrams() const {
  return this->m_description_bigrams;
}

inline const Section& AbstractBugReport::get_description_trigrams() const {
  return this->m_description_trigrams;
}

inline const Section& AbstractBugReport::get_all_unigrams() const {
  return this->m_all_unigrams;
}

inline const Section& AbstractBugReport::get_all_bigrams() const {
  return this->m_all_bigrams;
}

inline const Section& AbstractBugReport::get_all_trigrams() const {
  return this->m_all_trigrams;
}

inline unsigned AbstractBugReport::get_timestamp_in_days() const {
  return this->m_timestamp_in_days;
}

#endif /* _BUG_REPORT_H_ */
