/*
 * Postings.h
 *
 *  Created on: Mar 8, 2013
 *      Author: neo
 */

#ifndef _POSTINGS_H_
#define _POSTINGS_H_

#include <cassert>
#include <vector>

#include "TermInReport.h"

using std::vector;

class AbstractBugReport;

class Postings {
private:

  const int m_term_id;

  vector<TermInReport> m_reports;

  Postings(const Postings& copy);

  Postings& operator=(const Postings& copy);

public:

  const vector<TermInReport>& get_terms() const {
    return this->m_reports;
  }

  Postings(const int term_id) :
      m_term_id(term_id) {
  }

  int get_term_id() const {
    return this->m_term_id;
  }

  void add_report(const float normalized_tf, const AbstractBugReport& report);

  ~Postings();
};

#endif /* _POSTINGS_H_ */
