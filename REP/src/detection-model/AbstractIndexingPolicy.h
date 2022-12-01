/*
 * IndexingPolicy.h
 *
 *  Created on: Mar 9, 2013
 *      Author: neo
 */

#ifndef _ABSTRACT_INDEXING_POLICY_H_
#define _ABSTRACT_INDEXING_POLICY_H_

#include <vector>

//#include "../index/Master2BucketMap.h"
#include "../report-model/Term.h"
#include "IndexingType.h"

class AbstractBugReport;
class InvertedIndex;
class Master2PartialBucket;
class MasterBugReport;

using std::vector;

class AbstractIndexingPolicy {
private:

  InvertedIndex* m_index;

protected:

  AbstractIndexingPolicy(InvertedIndex* index);

  template<typename T>
  static void static_add_report(InvertedIndex* index,
      const vector<TermTemplate<T> >& terms, AbstractBugReport& bug_report);

  void add_report(const vector<Term>& terms,
      AbstractBugReport& bug_report) const;

  void add_report(const vector<PreciseTerm>& terms,
      AbstractBugReport& bug_report) const;

public:

  InvertedIndex* get_index() const {
    return m_index;
  }

  virtual IndexingType::EnumIndexingType get_report_candidates(
      const AbstractBugReport& query_report,
      Master2PartialBucket& candidate_collector) const = 0;

  virtual IndexingType::EnumIndexingType get_indexing_type() = 0;

  virtual ~AbstractIndexingPolicy();

  virtual void update_index(AbstractBugReport& bug_report) const = 0;

};

#endif /* _ABSTRACT_INDEXING_POLICY_H_ */
