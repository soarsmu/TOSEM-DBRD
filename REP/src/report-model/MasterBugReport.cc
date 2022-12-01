/*
 * MasterBugReport.cc
 *
 *  Created on: Jan 6, 2011
 *      Author: Chengnian SUN
 */

#include "MasterBugReport.h"
#include "DuplicateBugReport.h"
#include <cassert>
using namespace std;

bool MasterBugReport::is_detected() const {
  return false;
}

unsigned MasterBugReport::get_latest_timestamp_in_bucket() const {
  const unsigned dup_size = this->m_duplicates.size();
  if (dup_size) {
    return this->m_duplicates[dup_size - 1]->get_timestamp_in_days();
  } else {
    return this->get_timestamp_in_days();
  }
}

void MasterBugReport::set_detected() {
  // do nothing.
}

AbstractBugReport* MasterBugReport::get_copy() const {
  MasterBugReport* master = new MasterBugReport(*this);
  assert(master->m_duplicates.empty());
  return master;
}

void MasterBugReport::get_as_a_whole_bucket(
    vector<const AbstractBugReport*>& bucket_collector) const {
  assert(bucket_collector.empty());
  const unsigned size_of_duplicates = this->m_duplicates.size();
  bucket_collector.reserve(size_of_duplicates + 1);
  for (unsigned i = 0; i < size_of_duplicates; i++) {
    bucket_collector.push_back(this->m_duplicates[i]);
  }
  bucket_collector.push_back(this);
  assert(bucket_collector.size() <= (1 + this->m_duplicates.size()));
}

MasterBugReport::MasterBugReport(int id, const vector<Term>& summary_unigrams,
    const vector<Term>& summary_bigrams, const vector<Term>& summary_trigrams,
    const vector<Term>& descrption_unigrams,
    const vector<Term>& description_bigrams,
    const vector<Term>& description_trigrams, const vector<Term>& all_unigrams,
    const vector<Term>& all_bigrams, const vector<Term>& all_trigrams,
    const int version, const int component, const int sub_component,
    const int report_type, const int priority, const unsigned timestamp_in_days) :
    AbstractBugReport(id, id, summary_unigrams, summary_bigrams,
        summary_trigrams, descrption_unigrams, description_bigrams,
        description_trigrams, all_unigrams, all_bigrams, all_trigrams, version,
        component, sub_component, report_type, priority, timestamp_in_days) {
}

MasterBugReport::~MasterBugReport() {
}
