/*
 * IDFCollector.h
 *
 *  Created on: 2010-7-29
 *      Author: Chengnian Sun.
 */

#ifndef IDFCOLLECTOR_H_
#define IDFCOLLECTOR_H_

#include <vector>
#include <cmath>
#include <cassert>

using namespace std;


class IDFCollection {
private:

  class FrequencyInfo {
  private:
    float m_idf;

    unsigned m_number_of_docs_containing_the_term;

    int m_version;

  public:

    float get_idf(const int latest_version);

    void increase_frequency();

    FrequencyInfo();

  };

  vector<FrequencyInfo> m_frequencies;

  unsigned m_number_of_documents;

public:

  void add_one_report(const vector<Term>& unigrams,
      const vector<Term>& bigrams);

  /**
   * if ther term has no weight in the current idf collection, then return 0;
   */
  float get_idf(const int term_id);

  unsigned number_of_documents() const;

  IDFCollection(int max_id);

};

inline void IDFCollection::FrequencyInfo::increase_frequency() {
  this->m_number_of_docs_containing_the_term++;
}

inline IDFCollection::FrequencyInfo::FrequencyInfo() :
    m_idf(0), m_number_of_docs_containing_the_term(0), m_version(0) {
}

inline float IDFCollection::get_idf(const int term_id) {
  return this->m_frequencies[term_id].get_idf(this->m_number_of_documents);
}

inline unsigned IDFCollection::number_of_documents() const {
  return this->m_number_of_documents;
}

inline IDFCollection::IDFCollection(int max_id) {
  this->m_frequencies.assign(max_id + 1, FrequencyInfo());
  this->m_number_of_documents = 0;
}

#endif /* IDFCOLLECTOR_H_ */
