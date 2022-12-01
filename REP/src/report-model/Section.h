/*
 * SectionAndTerm.h
 *
 *  Created on: Feb 1, 2011
 *      Author: neo
 */

#ifndef SECTION_H__
#define SECTION_H__

#include <vector>
using namespace std;

#include "Term.h"

class Section {
private:

  const vector<Term> m_terms;

  const unsigned m_length;

  static unsigned compute_length(const vector<Term>& terms);

  Section(const vector<Term>& terms);

  friend class AbstractBugReport;

public:

  const vector<Term>& get_terms() const;

  unsigned get_length() const;

  double get_vector_length() const;

  ~Section();
};

class StructuredSection {
private:

  const vector<StructuredTerm> m_terms;

  const unsigned m_summary_length;

  const unsigned m_description_length;

  static vector<StructuredTerm> compute_terms(const Section& summary_section,
      const Section& description_section);

  friend class AbstractBugReport;

public:

  StructuredSection(const Section& summary_section,
      const Section& description_section);

  const vector<StructuredTerm>& get_terms() const;

  unsigned get_summary_length() const;

  unsigned get_description_length() const;
};

///////////////////////////////////////////////////////////////////////////////

inline const vector<Term>& Section::get_terms() const {
  return this->m_terms;
}

inline unsigned Section::get_length() const {
  return this->m_length;
}

inline Section::~Section() {
}

inline StructuredSection::StructuredSection(const Section& summary_section,
    const Section& description_section) :
    m_terms(compute_terms(summary_section, description_section)), m_summary_length(
        summary_section.get_length()), m_description_length(
        description_section.get_length()) {

}

inline const vector<StructuredTerm>& StructuredSection::get_terms() const {
  return this->m_terms;
}

inline unsigned StructuredSection::get_summary_length() const {
  return this->m_summary_length;
}

inline unsigned StructuredSection::get_description_length() const {
  return this->m_description_length;
}

#endif /* SECTION_H__ */
