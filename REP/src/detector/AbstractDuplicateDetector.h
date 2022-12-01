/*
 * DuplicateDetector.h
 *
 *  Created on: 2010-7-29
 *      Author: Chengnian Sun.
 */

#ifndef ABSTRACTDUPLICATEDETECTOR_H_
#define ABSTRACTDUPLICATEDETECTOR_H_

#include <cstdio>
#include <vector>

#include "../detection-model/IndexingType.h"

using namespace std;

class DetectionReportHistoryList;
class IDetectorExtension;
class ReportBuckets;
class AbstractToppingAlgorithm;
class MasterBugReport;
class DuplicateBugReport;
class AbstractBugReport;
class ReportDataset;

/**
 * 
 */
class AbstractDuplicateDetector {
private:

  FILE* m_log_file;

  // the size of result list.
  unsigned top_number;

  // this is to simulate the bug reporting activity
  DetectionReportHistoryList* report_list;

  // this count is the count of duplicate bug reports to skip.
  unsigned count_to_skip;

  // detector extensions. such as profiling, result calculator.
  vector<IDetectorExtension*> extensions;

  // representing report repository, it also includes the idf value collections.
  ReportBuckets* buckets;

  // the retrieval algorithm.
  AbstractToppingAlgorithm* topping_algorithm;

  vector<const MasterBugReport*> m_top_master_cache;

  // retrieving duplicate reports for the new duplicate.
  void handle_new_bug_report(AbstractBugReport& duplicate);

  bool m_detecting_all_reports;

protected:

  inline FILE* get_log_file() {
    return this->m_log_file;
  }

  // this will be called before detecting a new duplicate report.
  //virtual void pre_detection();

  // this will be called after detecting a new duplicate report.
  virtual void post_detection(AbstractBugReport& query_report,
      const vector<const MasterBugReport*>& top);

  // create the retrieval algorithm
  virtual AbstractToppingAlgorithm* create_topping_algorithm(
      const ReportBuckets& buckets) = 0;

  // this is the comment shown at the top of analysis file
  virtual const char* get_comment();

  virtual void log_detector_summary(FILE* log_file) = 0;

  // this is called after a new report comes. (can be duplicate or not). This will be called for every new report.
  void post_handle_report(AbstractBugReport* report);

  // skip count_to_skip bug reports
  virtual void skip();

public:
  AbstractDuplicateDetector(FILE* log_file, unsigned top_number,
      const ReportDataset& report_dataset, const unsigned count_to_skip,
      const vector<IDetectorExtension*>& extensions,
      const bool detecting_all_reports,
      const IndexingType::EnumIndexingType indexing_type);

  void init();

  void detect();

  virtual void dispose();

  virtual ~AbstractDuplicateDetector();
};

#endif /* ABSTRACTDUPLICATEDETECTOR_H_ */
