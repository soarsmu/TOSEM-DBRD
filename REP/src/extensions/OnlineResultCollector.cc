#include "OnlineResultCollector.h"
#include <iostream>
#include <cstdio>
#include <cassert>
#include <iomanip>
#include <utility>

#include "../report-model/MasterBugReport.h"
#include "../report-model/DuplicateBugReport.h"
#include "../util/PerformanceMeasureCalculator.h"

using namespace std;

OnlineResultCollector::OnlineResultCollector(unsigned int size_of_top_list,
    FILE* output_file) :
    m_size_of_top_list(size_of_top_list), m_output_file(output_file), m_column(
        0) {
  this->number_of_duplicates = 0;
  this->detection_start_time_per_duplicate = 0;
  this->current_duplicate_report = NULL;
  result_vector.reserve(600);
}
vector<std::pair<double, int> > OnlineResultCollector::get_recall_result() const {
  return PerformanceMeasureCalculator::compute_recall(this->m_size_of_top_list,
      this->result_vector);
}

void OnlineResultCollector::start_processing(const AbstractBugReport& report) {
  this->current_duplicate_report = &report;
  this->detection_start_time_per_duplicate = time(NULL);
  this->number_of_duplicates++;
}

void OnlineResultCollector::set_comment(const char*) {

}

//static int column = 0;
void OnlineResultCollector::handle_result(
    const vector<const MasterBugReport*>& candidates) {

  const time_t time_elapsed = time(NULL)
      - this->detection_start_time_per_duplicate;

  const int master_report_id =
      this->current_duplicate_report->get_duplicate_id();
  const int duplicate_report_id = this->current_duplicate_report->get_id();

//  printf("%4d ", this->number_of_duplicates);
//  this->m_column += 5;
//  if (this->m_column > 150) {
//    this->m_column = 0;
//    printf("\n");
//  }
  if (this->number_of_duplicates % 20 == 0) {
    printf(".");
    this->m_column++;
    if (this->m_column > 79) {
      this->m_column = 0;
      printf("%7d\n", this->number_of_duplicates);
    }
    fflush(stdout);
  }

  const unsigned int candidate_size = candidates.size();
  unsigned index =
      DetectionResultOfEachDuplicate::INVALID_INDEX_OF_MASTER_DETECTED;
  for (unsigned int i = 0; i < candidate_size; i++) {
    if (candidates[i]->get_id() == master_report_id) {
      index = i;
      break;
    }
  }

  this->result_vector.push_back(
      DetectionResultOfEachDuplicate(this->number_of_duplicates,
          duplicate_report_id, master_report_id, index, time_elapsed));

}
double OnlineResultCollector::get_mean_average_precision() const {
  return PerformanceMeasureCalculator::compute_mean_average_precision(
      this->result_vector);
}

void OnlineResultCollector::dispose() {

}

OnlineResultCollector::~OnlineResultCollector(void) {
}
