/*
 * DuplicateBugReport.h
 *
 *  Created on: 2010-7-29
 *      Author: Chengnian Sun
 */

#ifndef DUPLICATEBUGREPORT_H_
#define DUPLICATEBUGREPORT_H_

#include "AbstractBugReport.h"
class MasterBugReport;

class DuplicateBugReport: public AbstractBugReport {
private:

  bool m_detected;

  const MasterBugReport* m_master;

public:

  virtual AbstractBugReport* get_copy() const;

  virtual const MasterBugReport* get_master() const {
    return this->m_master;
  }

  virtual void set_detected();

  virtual bool is_detected() const;

  void set_master(const MasterBugReport* master);

  DuplicateBugReport(const int id, const int duplicate_id,
      const vector<Term>& summary_unigrams, const vector<Term>& summary_bigrams,
      const vector<Term>& summary_trigrams,
      const vector<Term>& descrption_unigrams,
      const vector<Term>& description_bigrams,
      const vector<Term>& description_trigrams,
      const vector<Term>& all_unigrams, const vector<Term>& all_bigrams,
      const vector<Term>& all_trigrams, const int version, const int component,
      const int sub_component, const int report_type, const int priority,
      const unsigned timestamp_in_days);

  virtual ~DuplicateBugReport();

};

#endif /* DUPLICATEBUGREPORT_H_ */
