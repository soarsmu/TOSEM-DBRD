/*
 * InvertedIndex.cc
 *
 *  Created on: Mar 8, 2013
 *      Author: neo
 */

#include <boost/unordered_map.hpp>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../report-model/AbstractBugReport.h"
#include "../report-model/DuplicateBugReport.h"
#include "../report-model/MasterBugReport.h"
#include "../report-model/Term.h"
#include "../tfidf/IDFCollection.h"
#include "../util/CmdOption.h"
#include "InvertedIndex.h"
#include "Master2PartialBucket.h"
#include "Postings.h"

using boost::unordered_map;

InvertedIndex::InvertedIndex(int max_term_id, IDFCollection* idf_collection) :
    m_index(static_cast<size_t>(max_term_id + 1), static_cast<Postings*>(NULL)), m_idf_threshold(
        CmdOption::get_instance().get_indexing_idf_threshold()), m_idf_collection(
        idf_collection) {
  assert(m_idf_collection);
}

InvertedIndex::~InvertedIndex() {
  for (vector<Postings*>::iterator iter = this->m_index.begin(), end =
      this->m_index.end(); iter != end; ++iter) {
    Postings* postings = *iter;
    if (postings)
      delete postings;
    *iter = NULL;
  }
}

Postings* InvertedIndex::get_postings(const int term_id) {
  Postings* postings = this->m_index[term_id];
  if (!postings) {
    postings = new Postings(term_id);
    this->m_index[term_id] = postings;
  }
  assert(postings->get_term_id() == term_id);
  return postings;
}

void InvertedIndex::get_reports(const AbstractBugReport& query_report,
    Master2PartialBucket& candidate_collector) {
  assert(candidate_collector.empty());
  const vector<Term>& terms = query_report.get_all_unigrams().get_terms();
  for (vector<Term>::const_iterator iter = terms.begin(), end = terms.end();
      iter != end; ++iter) {
    const Term& term = *iter;
    const int term_id = term.get_tid();
    const double idf = this->m_idf_collection->get_idf(term_id);

    if (idf < this->m_idf_threshold)
      continue;

    const Postings& postings = *this->get_postings(term.get_tid());

    const vector<TermInReport>& reports = postings.get_terms();
    for (vector<TermInReport>::const_iterator riter = reports.begin(), rend =
        reports.end(); riter != rend; ++riter) {
      const TermInReport& term_in_report = *riter;
      const AbstractBugReport* bug_report = term_in_report.get_report();
      const MasterBugReport* master;
      if (bug_report->is_duplicate()) {
        master =
            static_cast<const DuplicateBugReport*>(bug_report)->get_master();
      } else {
        master = static_cast<const MasterBugReport*>(bug_report);
      }
      candidate_collector.add(master->get_id(), bug_report,
          term_in_report.get_normalized_tf());
    }
  }

  candidate_collector.retrieval_done(query_report);
}
//
//template<typename T>
//void InvertedIndex::add_report(const vector<PreciseTerm>& normalized_terms,
//    AbstractBugReport& bug_report) {
//  T square_sum = 0;
//  for (vector<Term>::const_iterator iter = unnormalized_terms.begin(), end =
//      unnormalized_terms.end(); iter != end; ++iter) {
//    const Term& term = *iter;
//    const int tf = term.get_term_frequency();
//    square_sum += tf * tf;
//  }
//  const float vector_length = std::sqrt(square_sum);
//  for (vector<Term>::const_iterator iter = unnormalized_terms.begin(), end =
//      unnormalized_terms.end(); iter != end; ++iter) {
//    const Term& term = *iter;
//    Postings* postings = this->get_postings(term.get_tid());
//    postings->add_report(term.get_tid() / vector_length, bug_report);
//  }
//}
void InvertedIndex::add_report(const vector<PreciseTerm>& normalized_terms,
    const AbstractBugReport& bug_report) {
  for (vector<PreciseTerm>::const_iterator iter = normalized_terms.begin(),
      end = normalized_terms.end(); iter != end; ++iter) {
    const PreciseTerm& term = *iter;
    Postings* postings = this->get_postings(term.get_tid());
    postings->add_report(term.get_term_frequency(), bug_report);
  }
}

