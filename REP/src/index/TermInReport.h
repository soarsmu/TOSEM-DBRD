/*
 * TermInReport.h
 *
 *  Created on: Mar 13, 2013
 *      Author: Chengnian Sun
 */

#ifndef _TERM_IN_REPORT_H_
#define _TERM_IN_REPORT_H_

class AbstractBugReport;

class TermInReport {
private:

  const AbstractBugReport* m_report;

  float m_normalized_term_frequency;

public:

  TermInReport(const AbstractBugReport* bug_report,
      float normalized_term_frequency);

  float get_normalized_tf() const;

  const AbstractBugReport* get_report() const;

};

///////////////////////////////////////////////////////////////////////////////

inline TermInReport::TermInReport(const AbstractBugReport* bug_report,
    float normalized_term_frequency) :
    m_report(bug_report), m_normalized_term_frequency(normalized_term_frequency) {
}

inline float TermInReport::get_normalized_tf() const {
  return this->m_normalized_term_frequency;
}

inline const AbstractBugReport* TermInReport::get_report() const {
  return this->m_report;
}

#endif /* _TERM_IN_REPORT_H_ */
