/*
 * NoIndexingPolicy.h
 *
 *  Created on: Mar 9, 2013
 *      Author: neo
 */

#ifndef NOINDEXINGPOLICY_H_
#define NOINDEXINGPOLICY_H_

#include "AbstractIndexingPolicy.h"

class NoIndexingPolicy: public AbstractIndexingPolicy {
public:

  NoIndexingPolicy();

  virtual ~NoIndexingPolicy();

  virtual void update_index(AbstractBugReport& bug_report) const;

  virtual IndexingType::EnumIndexingType get_report_candidates(
      const AbstractBugReport&, Master2PartialBucket&) const {
    return IndexingType::NO_INDEXING;
  }

  virtual IndexingType::EnumIndexingType get_indexing_type() {
    return IndexingType::NO_INDEXING;
  }
};

#endif /* NOINDEXINGPOLICY_H_ */
