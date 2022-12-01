/*
 * IDFCollection.cc
 *
 *  Created on: Mar 8, 2013
 *      Author: neo
 */

#include <cmath>
#include "../report-model/Term.h"
#include "IDFCollection.h"

const float LOG_2 = log(2);

float IDFCollection::FrequencyInfo::get_idf(const int latest_version) {
  if (this->m_number_of_docs_containing_the_term == 0) {
    return 0;
  } else {
    if (this->m_version < latest_version) {
      this->m_version = latest_version;
      this->m_idf = log(
          static_cast<float>(latest_version)
              / this->m_number_of_docs_containing_the_term) / LOG_2;
    }
    return this->m_idf;
  }
}

void IDFCollection::add_one_report(const vector<Term>& unigrams,
    const vector<Term>& bigrams) {

  for (vector<Term>::const_iterator iter = unigrams.begin(), end =
      unigrams.end(); iter != end; ++iter) {
    this->m_frequencies[iter->get_tid()].increase_frequency();
  }

  for (vector<Term>::const_iterator iter = bigrams.begin(), end = bigrams.end();
      iter != end; ++iter) {
    this->m_frequencies[iter->get_tid()].increase_frequency();
  }

  ++this->m_number_of_documents;

}
