/*
 * InvertedIndex.h
 *
 *  Created on: Mar 8, 2013
 *      Author: neo
 */

#ifndef _INVERTED_INDEX_H__
#define _INVERTED_INDEX_H__

#include <vector>

//#include "../index/Master2BucketMap.h"
#include "../report-model/Term.h"

using std::vector;

class AbstractBugReport;
class IDFCollection;
class Master2PartialBucket;
class Postings;

class InvertedIndex {
  friend class AbstractIndexingPolicy;
  friend class IndexedRankNetToppingAlgorithm;
private:

  vector<Postings*> m_index;

  const double m_idf_threshold;

  IDFCollection* m_idf_collection;

  void add_report(const vector<PreciseTerm>& normalized_terms,
      const AbstractBugReport& bug_report);

//  void add_report(const vector<Term>& unnormalized_terms,
//      AbstractBugReport& bug_report);
//
//  void add_report(const vector<PreciseTerm>& unnormalized_terms,
//      AbstractBugReport& bug_report);

public:

  Postings* get_postings(const int term_id);

  void get_reports(const AbstractBugReport& query_report,
      Master2PartialBucket& candidate_collector);

  InvertedIndex(int max_term_id, IDFCollection* idf_collection);

  ~InvertedIndex();

};

#endif /* _INVERTED_INDEX_H__ */
