/*
 * CommonIndexingPolicy.h
 *
 *  Created on: Mar 9, 2013
 *      Author: neo
 */

#ifndef _COMMON_INDEXING_POLICY_H_
#define _COMMON_INDEXING_POLICY_H_

#include "AbstractIndexingPolicy.h"

class IDFCollection;

class CommonIndexingPolicy: public AbstractIndexingPolicy {
private:

  const IndexingType::EnumIndexingType m_type;

  const unsigned m_summary_weight;

public:

  CommonIndexingPolicy(int max_term_id, IndexingType::EnumIndexingType type,
      IDFCollection* idf_collection);

  virtual ~CommonIndexingPolicy();

  virtual IndexingType::EnumIndexingType get_report_candidates(
      const AbstractBugReport& query_report,
      Master2PartialBucket& candidate_collector) const;

  virtual IndexingType::EnumIndexingType get_indexing_type() {
    return this->m_type;
  }

  virtual void update_index(AbstractBugReport& bug_report) const;

};

#endif /* _COMMON_INDEXING_POLICY_H_ */
