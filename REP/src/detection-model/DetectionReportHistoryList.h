/*
 * DetectionReportList.h
 *
 *  Created on: 2010-7-30
 *      Author: Chengnian Sun.
 */

#ifndef DETECTIONREPORTLIST_H_
#define DETECTIONREPORTLIST_H_

#include <vector>
using namespace std;

class AbstractBugReport;
class ReportDataset;

class DetectionReportHistoryList {
private:

  const vector<AbstractBugReport*> * reports;

  unsigned int current_report_index;

  int max_term_id;

  AbstractBugReport* current_report;

  vector<AbstractBugReport*> visited_duplicates;

public:
  DetectionReportHistoryList(const ReportDataset& report_dataset);

  int report_count() const;

  bool has_report();

  int get_max_term_id() const;

  AbstractBugReport* next_report();

  ~DetectionReportHistoryList();
};

inline int DetectionReportHistoryList::report_count() const {
  return this->reports->size();
}

inline bool DetectionReportHistoryList::has_report() {
  return this->current_report_index < this->reports->size();
}

inline int DetectionReportHistoryList::get_max_term_id() const {
  return this->max_term_id;
}

#endif /* DETECTIONREPORTLIST_H_ */
