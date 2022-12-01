/*
 * RecallAndTimeIntervalAnalysis.cc
 *
 *  Created on: Apr 8, 2013
 *      Author: neo
 */

#include <cstdlib>
#include <cstddef>

#include "../report-model/DuplicateBugReport.h"
#include "../report-model/MasterBugReport.h"
#include "../util/MacroUtility.h"
#include "RecallAndTimeIntervalAnalysis.h"

RecallAndTimeIntervalAnalysis::RecallAndTimeIntervalAnalysis(
    unsigned size_of_top_list, FILE* output_file) :
    m_size_of_top_list(size_of_top_list), m_output_file(output_file), current_new_report(
        NULL) {
  if (!output_file) {
    ERROR_HERE("The argument output_file cannot be null");
  }
}

void RecallAndTimeIntervalAnalysis::start_processing(
    const AbstractBugReport& report) {
  assert(this->m_output_file);
  this->current_new_report = report.is_duplicate() ? &report : NULL;
//  fprintf(this->m_output_file, "%6s@%2s \tinterval=%-4s\n", "dup", "index",
//      "delta");
}

void RecallAndTimeIntervalAnalysis::handle_result(
    const vector<const MasterBugReport*>& candidates) {
  if (!this->current_new_report) {
    return;
  }

  const int dup_id = this->current_new_report->get_duplicate_id();

  int detected_index = -1;
  const MasterBugReport* master = NULL;

  for (unsigned i = 0, size = candidates.size();
      i < size && i < this->m_size_of_top_list; ++i) {
    const MasterBugReport* can_report = candidates[i];
    if (can_report->get_id() == dup_id) {
      detected_index = i;
      master = can_report;
      break;
    }
  }

  if (detected_index > -1) {
    assert(master);
    fprintf(this->m_output_file, "%6d@%2d \t interval=%4u\n",
        this->current_new_report->get_id(), detected_index,
        (this->current_new_report->get_timestamp_in_days()
            - master->get_latest_timestamp_in_bucket()));
  }
}

void RecallAndTimeIntervalAnalysis::dispose() {

}

void RecallAndTimeIntervalAnalysis::set_comment(const char*) {
}

RecallAndTimeIntervalAnalysis::~RecallAndTimeIntervalAnalysis() {
}

