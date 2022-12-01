/*
 * DetectionReportList.cpp
 *
 *  Created on: 2010-7-30
 *      Author: Chengnian Sun.
 */

#include "DetectionReportHistoryList.h"
#include "../report-model/AbstractBugReport.h"
#include "../report-model/ReportDataset.h"
#include <cassert>
#include <cstdio>
using namespace std;

AbstractBugReport* DetectionReportHistoryList::next_report() {
  AbstractBugReport* report = this->reports->at(this->current_report_index++);
  this->current_report = report;
  if (report->is_duplicate()) {
    this->visited_duplicates.push_back(report);
  }
  return report;
}

DetectionReportHistoryList::DetectionReportHistoryList(
    const ReportDataset& report_dataset) {
  vector<AbstractBugReport*>* report_collector =
      new vector<AbstractBugReport*>();

  //	ReportReader reader(report_dataset_path, report_collector);
  report_dataset.get_copy(*report_collector);

  this->reports = report_collector;

  this->max_term_id = report_dataset.get_max_term_id();

  this->current_report_index = 0;
  this->current_report = NULL;

}

DetectionReportHistoryList::~DetectionReportHistoryList() {
  const unsigned size = this->reports->size();
  for (unsigned i = 0; i < size; i++) {
    delete this->reports->at(i);
  }
  delete this->reports;
  this->reports = NULL;
}
