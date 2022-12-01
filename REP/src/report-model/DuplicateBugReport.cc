/*
 * DuplicateBugReport.cc
 *
 *  Created on: Jan 14, 2011
 *      Author: Chengnian SUN
 */

#include "DuplicateBugReport.h"
#include <cassert>
#include "MasterBugReport.h"

DuplicateBugReport::~DuplicateBugReport() {

}

bool DuplicateBugReport::is_detected() const {
  return this->m_detected;
}

void DuplicateBugReport::set_detected() {
  this->m_detected = true;
}

DuplicateBugReport::DuplicateBugReport(const int id, const int duplicate_id,
    const vector<Term>& summary_unigrams, const vector<Term>& summary_bigrams,
    const vector<Term>& summary_trigrams,
    const vector<Term>& descrption_unigrams,
    const vector<Term>& description_bigrams,
    const vector<Term>& description_trigrams, const vector<Term>& all_unigrams,
    const vector<Term>& all_bigrams, const vector<Term>& all_trigrams,
    const int version, const int component, const int sub_component,
    const int report_type, const int priority, const unsigned timestamp_in_days) :
    AbstractBugReport(id, duplicate_id, summary_unigrams, summary_bigrams,
        summary_trigrams, descrption_unigrams, description_bigrams,
        description_trigrams, all_unigrams, all_bigrams, all_trigrams, version,
        component, sub_component, report_type, priority, timestamp_in_days) {
  this->m_detected = false;
  this->m_master = NULL;
}

void DuplicateBugReport::set_master(const MasterBugReport* master) {
  assert(master != NULL && master->get_id() == this->get_duplicate_id());
  this->m_master = master;
}

AbstractBugReport* DuplicateBugReport::get_copy() const {
  DuplicateBugReport* dup = new DuplicateBugReport(*this);
  assert(dup->m_master == NULL);
  assert(dup->m_detected == false);
  return dup;
}

