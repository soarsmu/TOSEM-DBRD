/*
 * Master2PartialBucket.h
 *
 *  Created on: Mar 13, 2013
 *      Author: Chengnian Sun
 */

#ifndef MASTER2PARTIALBUCKET_H_
#define MASTER2PARTIALBUCKET_H_

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <vector>

using boost::unordered_map;
using boost::unordered_set;
using std::vector;

class AbstractBugReport;

class Master2PartialBucket {
public:

  class ReportWithSimilarity {
  private:

    const AbstractBugReport* m_report;

    float m_similarity;

  public:

    ReportWithSimilarity(const AbstractBugReport* report,
        unsigned init_similarity);

    void increase_similarity(const float delta);

    const AbstractBugReport* get_report() const;

    float get_similarity() const;

  };

  typedef int MasterID;

  typedef vector<ReportWithSimilarity> PartialBucket;

  typedef boost::unordered_map<MasterID, PartialBucket*> BucketMap;

  typedef BucketMap::const_iterator MapIterator;

private:

  BucketMap m_buckets;

  vector<PartialBucket *> m_used;

  vector<PartialBucket *> m_unused;

  boost::unordered_set<int> m_accepted;

  const unsigned m_top_cosine_number;

  unsigned m_number_of_reports;

  PartialBucket* get_or_create_bucket();

  void retrieval_done(const AbstractBugReport& query_report);

  friend class InvertedIndex;

public:

  MapIterator begin() const;

  MapIterator end() const;

  void add(MasterID master_id, const AbstractBugReport* report,
      const float similarity_delta);

  unsigned number_of_reports() const;

  void clear();

  bool filtered(const int report_id) const;

  bool empty() const;

  size_t size() const;

  Master2PartialBucket(unsigned top_cosine_number);

  ~Master2PartialBucket();
};

///////////////////////////////////////////////////////////////////////////////

inline unsigned Master2PartialBucket::number_of_reports() const {
  return this->m_number_of_reports;
}

inline bool Master2PartialBucket::filtered(const int report_id) const {
  return this->m_top_cosine_number && this->m_accepted.count(report_id) == 0;
}

inline Master2PartialBucket::ReportWithSimilarity::ReportWithSimilarity(
    const AbstractBugReport* report, unsigned init_similarity) :
    m_report(report), m_similarity(init_similarity) {
}

inline void Master2PartialBucket::ReportWithSimilarity::increase_similarity(
    const float delta) {
  this->m_similarity += delta;
}

inline const AbstractBugReport* Master2PartialBucket::ReportWithSimilarity::get_report() const {
  return this->m_report;
}

inline float Master2PartialBucket::ReportWithSimilarity::get_similarity() const {
  return this->m_similarity;
}

inline Master2PartialBucket::MapIterator Master2PartialBucket::begin() const {
  return this->m_buckets.begin();
}

inline Master2PartialBucket::MapIterator Master2PartialBucket::end() const {
  return this->m_buckets.end();
}

inline bool Master2PartialBucket::empty() const {
  return this->m_buckets.empty();
}

inline size_t Master2PartialBucket::size() const {
  return this->m_buckets.size();
}

#endif /* MASTER2PARTIALBUCKET_H_ */
