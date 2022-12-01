/*
 * MasterBugReport.h
 *
 *  Created on: 2010-7-29
 *      Author: Chengnian Sun
 */

#ifndef MASTERBUGREPORT_H_
#define MASTERBUGREPORT_H_

#include <algorithm>
#include <vector>
using namespace std;

#include "AbstractBugReport.h"

class DuplicateBugReport;

class MasterBugReport: public AbstractBugReport {
private:
  vector<const DuplicateBugReport*> m_duplicates;

public:

  unsigned get_latest_timestamp_in_bucket() const;

  virtual const MasterBugReport* get_master() const;

  virtual void set_detected();

  virtual bool is_detected() const;

  virtual AbstractBugReport* get_copy() const;

  const vector<const DuplicateBugReport*>& get_duplicates() const;

  /**
   * return all the duplicates and the master.
   */
  void get_as_a_whole_bucket(
      vector<const AbstractBugReport*>& bucket_collector) const;

  void add_duplicate(const DuplicateBugReport* duplicate);

  bool has_duplicates() const;

  bool has_no_duplicates() const;

  MasterBugReport(int id, const vector<Term>& summary_unigrams,
      const vector<Term>& summary_bigrams, const vector<Term>& summary_trigrams,
      const vector<Term>& descrption_unigrams,
      const vector<Term>& description_bigrams,
      const vector<Term>& description_trigrams,
      const vector<Term>& all_unigrams, const vector<Term>& all_bigrams,
      const vector<Term>& all_trigrams, const int version, const int component,
      const int sub_component, const int report_type, const int priority,
      const unsigned timestamp_in_days);

  virtual ~MasterBugReport();

};

inline void MasterBugReport::add_duplicate(
    const DuplicateBugReport* duplicate) {
  this->m_duplicates.push_back(duplicate);
}

inline bool MasterBugReport::has_duplicates() const {
  return !this->m_duplicates.empty();
}

inline bool MasterBugReport::has_no_duplicates() const {
  return !this->has_duplicates();
}

inline const vector<const DuplicateBugReport*>&
MasterBugReport::get_duplicates() const {
  return this->m_duplicates;
}

inline const MasterBugReport* MasterBugReport::get_master() const {
  return this;
}

#endif /* MASTERBUGREPORT_H_ */
