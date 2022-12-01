/*
 * ReportDataset.cc
 *
 *  Created on: Jan 14, 2011
 *      Author: neo
 */

#include <cassert>
#include <cstdio>
#include <string>

#include "ReportReader.h"
#include "ReportDataset.h"
#include "AbstractBugReport.h"

using namespace std;

void ReportDataset::get_copy(vector<AbstractBugReport*>& copy_collector) const {
  assert(this->m_all_reports_in_dataset != NULL);
  assert(copy_collector.empty());

  for (vector<AbstractBugReport*>::iterator iter =
      this->m_all_reports_in_dataset->begin();
      iter != this->m_all_reports_in_dataset->end(); iter++) {
    AbstractBugReport* report = *iter;
    copy_collector.push_back(report->get_copy());
  }
}

ReportDataset::ReportDataset(const string& report_dataset_path,
    const string& timestamp_file) {
  printf("reading reports from %s...\n", report_dataset_path.c_str());
  printf("reading time-stamps from %s...\n", timestamp_file.c_str());
  vector<AbstractBugReport*>* report_collector =
      new vector<AbstractBugReport*>();
  ReportReader reader(report_dataset_path, timestamp_file, report_collector);

  this->m_all_reports_in_dataset = report_collector;
  this->m_max_term_id = reader.get_max_term_id();
}

ReportDataset::~ReportDataset() {
  assert(this->m_all_reports_in_dataset);
  for (vector<AbstractBugReport*>::iterator iter =
      this->m_all_reports_in_dataset->begin();
      iter != this->m_all_reports_in_dataset->end(); iter++) {
    delete *iter;
  }
  delete this->m_all_reports_in_dataset;
  this->m_all_reports_in_dataset = NULL;
}
