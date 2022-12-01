/*
 * IPlainSimilarityMeasure.h
 *
 *  Created on: Dec 11, 2010
 *      Author: Chengnian SUN
 */

#ifndef IPLAINSIMILARITYMEASURE_H_
#define IPLAINSIMILARITYMEASURE_H_

#include <boost/unordered_map.hpp>

#include "../detection-model/ReportBuckets.h"
#include "../report-model/MasterBugReport.h"
#include "../report-model/DuplicateBugReport.h"

using namespace boost;

class IPlainSimilarityMeasure {

protected:

public:

  virtual double compute_similarity(const AbstractBugReport& query_report,
      const AbstractBugReport& doc_report, const ReportBuckets& bucket) = 0;

  IPlainSimilarityMeasure() {
  }

  virtual ~IPlainSimilarityMeasure() {

  }
};

#endif /* IPLAINSIMILARITYMEASURE_H_ */
