/*
 * CommonIndexingPolicy.cc
 *
 *  Created on: Mar 9, 2013
 *      Author: neo
 */
#include <algorithm>
#include <cassert>

#include "../index/InvertedIndex.h"
#include "../report-model/AbstractBugReport.h"
#include "../report-model/Term.h"
#include "../util/CmdOption.h"
#include "../util/MacroUtility.h"
#include "CommonIndexingPolicy.h"

CommonIndexingPolicy::CommonIndexingPolicy(int max_term_id,
    IndexingType::EnumIndexingType type, IDFCollection* idf_collection) :
    AbstractIndexingPolicy(new InvertedIndex(max_term_id, idf_collection)), m_type(
        type), m_summary_weight(
        CmdOption::get_instance().indexing_summary_weight()) {
  assert(this->m_type != IndexingType::NO_INDEXING);
}

CommonIndexingPolicy::~CommonIndexingPolicy() {
}

IndexingType::EnumIndexingType CommonIndexingPolicy::get_report_candidates(
    const AbstractBugReport& query_report,
    Master2PartialBucket& candidate_collector) const {
//  const vector<Term>& terms = query_report.get_all_unigrams().get_terms();
  this->get_index()->get_reports(query_report, candidate_collector);
  return this->m_type;
}

static void compute_weighed_terms(vector<PreciseTerm>& weighed_terms_collector,
    const float summary_weight, AbstractBugReport& bug_report) {
  assert(weighed_terms_collector.empty());
  const vector<Term>& summary = bug_report.get_summary_unigrams().get_terms();
  const vector<Term>& desc = bug_report.get_description_unigrams().get_terms();
  const size_t sum_size = summary.size();
  const size_t desc_size = desc.size();
  weighed_terms_collector.reserve(sum_size + desc_size);

  size_t sum_i = 0;
  size_t desc_i = 0;
  while (sum_i < sum_size && desc_i < desc_size) {
    const Term& sum_term = summary[sum_i];
    const Term& desc_term = desc[desc_i];
    const int sum_term_id = sum_term.get_tid();
    const int desc_term_id = desc_term.get_tid();

    if (sum_term_id == desc_term_id) {
      weighed_terms_collector.push_back(
          PreciseTerm(sum_term_id,
              sum_term.get_term_frequency() * summary_weight
                  + desc_term.get_term_frequency()));
      ++sum_i;
      ++desc_i;
    } else if (sum_term_id > desc_term_id) {
      weighed_terms_collector.push_back(
          PreciseTerm(desc_term_id, desc_term.get_term_frequency()));
      ++desc_i;
    } else {
      weighed_terms_collector.push_back(
          PreciseTerm(sum_term_id,
              sum_term.get_term_frequency() * summary_weight));
      ++sum_i;
    }
  }
  while (sum_i < sum_size) {
    const Term& sum_term = summary[sum_i];
    weighed_terms_collector.push_back(
        PreciseTerm(sum_term.get_tid(),
            sum_term.get_term_frequency() * summary_weight));
    ++sum_i;
  }
  while (desc_i < desc_size) {
    const Term& desc_term = desc[desc_i];
    weighed_terms_collector.push_back(
        PreciseTerm(desc_term.get_tid(), desc_term.get_term_frequency()));
    ++desc_i;
  }
#ifndef NDEBUG
  assert(weighed_terms_collector.size() >= summary.size());
  assert(weighed_terms_collector.size() >= desc.size());
  for (size_t i = 1, size = weighed_terms_collector.size(); i < size; ++i) {
    assert(
        weighed_terms_collector[i - 1].get_tid() < weighed_terms_collector[i].get_tid());
  }
#endif
}

void CommonIndexingPolicy::update_index(AbstractBugReport& bug_report) const {
  switch (this->m_type) {
  case IndexingType::SUMMARY_INDEXING:
    this->add_report(bug_report.get_summary_unigrams().get_terms(), bug_report);
    break;
  case IndexingType::DESCRIPTION_INDEXING:
    this->add_report(bug_report.get_description_unigrams().get_terms(),
        bug_report);
    return;
  case IndexingType::FULL_INDEXING: {
    vector<PreciseTerm> weighed_terms;
    compute_weighed_terms(weighed_terms, 2.0, bug_report);
    assert(weighed_terms.size());
    this->add_report(weighed_terms, bug_report);
    return;
  }
  case IndexingType::NO_INDEXING:
  default: {
    UNREACHABLE(IndexingType::get_indexing_type_string(this->m_type).c_str());
    return;
  }
  }
}
