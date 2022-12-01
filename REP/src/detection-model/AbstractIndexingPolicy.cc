/*
 * IndexingPolicy.cc
 *
 *  Created on: Mar 9, 2013
 *      Author: neo
 */

#include <cmath>
#include <cstdlib>

#include "../index/InvertedIndex.h"
#include "AbstractIndexingPolicy.h"

AbstractIndexingPolicy::AbstractIndexingPolicy(InvertedIndex* index) :
    m_index(index) {

}

AbstractIndexingPolicy::~AbstractIndexingPolicy() {
  if (this->m_index) {
    delete this->m_index;
    this->m_index = NULL;
  }
}

template<typename T>
void AbstractIndexingPolicy::static_add_report(InvertedIndex* index,
    const vector<TermTemplate<T> >& unnormalized_terms,
    AbstractBugReport& bug_report) {
  T square_sum = 0;
  const unsigned size = unnormalized_terms.size();
  for (unsigned i = 0; i < size; ++i) {
    const TermTemplate<T>& term = unnormalized_terms[i];
    const T tf = term.get_term_frequency();
    square_sum += tf * tf;
  }
  const float vector_length = std::sqrt(square_sum);

  vector<PreciseTerm> normalized_terms;
  normalized_terms.reserve(size);
  for (unsigned i = 0; i < size; ++i) {
    const TermTemplate<T>& term = unnormalized_terms[i];
    normalized_terms.push_back(
        PreciseTerm(term.get_tid(), term.get_term_frequency() / vector_length));
  }
  index->add_report(normalized_terms, bug_report);
}

void AbstractIndexingPolicy::add_report(const vector<Term>& terms,
    AbstractBugReport& bug_report) const {
  AbstractIndexingPolicy::static_add_report(this->m_index, terms, bug_report);
}

void AbstractIndexingPolicy::add_report(const vector<PreciseTerm>& terms,
    AbstractBugReport& bug_report) const {
  AbstractIndexingPolicy::static_add_report(this->m_index, terms, bug_report);
}
