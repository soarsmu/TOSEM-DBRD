/*
 * RecommendationRecorder.cc
 *
 *  Created on: Aug 30, 2011
 *      Author: Chengnian Sun
 */

#include <cstdlib>
#include <cstddef>

#include "../report-model/DuplicateBugReport.h"
#include "../report-model/MasterBugReport.h"
#include "../util/MacroUtility.h"

#include "RecommendationRecorder.h"

RecommendationRecorder::RecommendationRecorder(unsigned size_of_top_list,
    FILE* output_file) :
    m_size_of_top_list(size_of_top_list), m_output_file(output_file), current_new_report(
        NULL) {
  if (!output_file) {
    ERROR_HERE("The argument output_file cannot be null");
  }
}

void RecommendationRecorder::start_processing(const AbstractBugReport& report) {
  this->current_new_report = &report;
  assert(this->m_output_file);
  if (report.is_duplicate()) {
    fprintf(this->m_output_file,
        "Retrieving for duplicate report %d (Its master is %d)\n",
        report.get_id(), report.get_duplicate_id());
  } else {
    fprintf(this->m_output_file, "Retrieving for non-duplicate report %d\n",
        report.get_id());
  }
}

void RecommendationRecorder::set_comment(const char*) {

}

void RecommendationRecorder::handle_result(
    const vector<const MasterBugReport*>& candidates) {
  const std::size_t size = candidates.size();
  const int dup_id = this->current_new_report->get_duplicate_id();
  for (unsigned i = 0; i < size && i < this->m_size_of_top_list; ++i) {
    const int can_id = candidates[i]->get_id();
    const int real_dup_id =
        candidates[i]->get_similarity_info()->get_similar_report_id();
    const double similarity =
        candidates[i]->get_similarity_info()->get_similarity();
    const char* sign = (can_id == dup_id ? "+" : " ");
    fprintf(this->m_output_file, "%2u - %6d(real-sim-id=%6d) : %-3.6f %s\n",
        (i + 1), can_id, real_dup_id, similarity, sign);
  }
  fprintf(this->m_output_file, "\n");
}

void RecommendationRecorder::dispose() {

}

RecommendationRecorder::~RecommendationRecorder() {
}

