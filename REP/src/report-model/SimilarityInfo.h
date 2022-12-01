/*
 * SimilarityInfo.h
 *
 *  Created on: Mar 12, 2013
 *      Author: neo
 */

#ifndef _SIMILARITY_INFO_H__
#define _SIMILARITY_INFO_H__

#include <cassert>
#include <utility>

using std::pair;

class SimilarityInfo {
private:
  double m_similarity;

  int m_similar_report_id;

public:

  void reset() {
    this->m_similar_report_id = 0;
    this->m_similarity = 0;
  }

  double get_similarity() const {
    return this->m_similarity;
  }

  int get_similar_report_id() const {
    return this->m_similar_report_id;
  }

  void set_similarity(const double similarity, const int similar_report_id) {
    this->m_similarity = similarity;
    this->m_similar_report_id = similar_report_id;
  }

  void set_similarity(const std::pair<int, double>& similarity_pair) {
    this->set_similarity(similarity_pair.second, similarity_pair.first);
  }
};

#endif /* SIMILARITYINFO_H_ */
