/*
 * DuplicateDetector.cpp
 *
 *  Created on: 2010-7-29
 *      Author: Chengnian Sun.
 */
#include <cassert>
#include <cstdio>
#include <string>

#include "../detection-model/DetectionReportHistoryList.h"
#include "../detection-model/ReportBuckets.h"
#include "../extensions/IDetectorExtension.h"
#include "../report-model/DuplicateBugReport.h"
#include "../report-model/MasterBugReport.h"
#include "../topping/AbstractToppingAlgorithm.h"
#include "../util/MacroUtility.h"
#include "AbstractDuplicateDetector.h"

using namespace std;

AbstractDuplicateDetector::AbstractDuplicateDetector(FILE* log_file,
    unsigned top_number, const ReportDataset& report_dataset,
    const unsigned count_to_skip, const vector<IDetectorExtension*>& extensions,
    const bool detecting_all_reports,
    const IndexingType::EnumIndexingType indexing_type) {

  this->m_log_file = log_file;

  this->m_detecting_all_reports = detecting_all_reports;

  //	TC_INFO(printf("reading reports from %s\n", report_dataset_path);)
  this->report_list = new DetectionReportHistoryList(report_dataset);
  if (this->m_log_file) {
    fprintf(this->m_log_file, "INFO: #reports = %u\n",
        this->report_list->report_count());
  }
  TC_INFO(printf("read %d reports\n", this->report_list->report_count());)
  this->extensions = extensions;

  TC_INFO(printf("creating report buckets with max term id %d\n", this->report_list->get_max_term_id());)
  this->buckets = new ReportBuckets(this->report_list->get_max_term_id(),
      indexing_type);

  this->count_to_skip = count_to_skip;
  this->top_number = top_number;
  this->topping_algorithm = NULL;
  this->m_top_master_cache.reserve(top_number);
}

void AbstractDuplicateDetector::skip() {
  TC_INFO(printf("skipping %d duplicates\n", this->count_to_skip);)
  if (this->count_to_skip <= 0) {
    return;
  }
  AbstractBugReport* report;
  while (this->report_list->has_report()) {
    report = this->report_list->next_report();
    this->post_handle_report(report);
    if (this->buckets->get_duplicates_count() >= this->count_to_skip) {
      return;
    }
  }

}

void AbstractDuplicateDetector::init() {
  //	TRACE(cout << "initializing detector...\n");
  TC_INFO(printf("initializing detector...\n");)

  this->topping_algorithm = this->create_topping_algorithm(*(this->buckets));

  assert(this->m_log_file);
  this->log_detector_summary(this->m_log_file);
}

const char* AbstractDuplicateDetector::get_comment() {
  return "";
}

void AbstractDuplicateDetector::dispose() {
  for (unsigned int i = 0; i < this->extensions.size(); ++i) {
    this->extensions[i]->dispose();
  }
}

void AbstractDuplicateDetector::detect() {
  //	TRACE(cout << "starting to simulate...\n");
  TC_INFO(printf("starting to simulate...\n");)
  //	TRACE(cout << "initialize extensions...\n");
  for (unsigned int i = 0; i < this->extensions.size(); i++) {
    this->extensions[i]->set_comment(this->get_comment());
  }
  this->skip();
  if (this->m_log_file) {
    fprintf(this->m_log_file, "INFO: skipped reports [(%u)duplicate/(%u)all]\n",
        this->count_to_skip, this->buckets->get_report_count());
    fprintf(stdout, "INFO: skipped reports [(%u)duplicate/(%u)all]\n",
        this->count_to_skip, this->buckets->get_report_count());
  }

  AbstractBugReport* report = NULL;
  int n_iteration = 0;

  while (this->report_list->has_report()) {
    report = this->report_list->next_report();
    if (this->m_detecting_all_reports || report->is_duplicate()) {
      this->handle_new_bug_report(*report);
      n_iteration++;
    }
    this->post_handle_report(report);
  }

  // Check whether the program entered in the loop. If not, we train the model.
  if(n_iteration == 0){
    this->topping_algorithm->train_model();
  }

}

void AbstractDuplicateDetector::post_detection(AbstractBugReport& query_report,
    const vector<const MasterBugReport*>& top) {
  for (unsigned int i = 0; i < top.size() && i < this->top_number; ++i) {
    if (top[i]->get_id() == query_report.get_duplicate_id()) {
      query_report.set_detected();
      return;
    }
  }
}

void AbstractDuplicateDetector::handle_new_bug_report(
    AbstractBugReport& query) {
  assert((!query.is_duplicate()) || query.get_id() > query.get_duplicate_id());

  for (unsigned int i = 0, size = this->extensions.size(); i < size; ++i) {
    this->extensions[i]->start_processing(query);
  }

  this->m_top_master_cache.clear();

  this->topping_algorithm->get_top(query, this->m_top_master_cache);

  this->post_detection(query, this->m_top_master_cache);
  for (unsigned int i = 0; i < this->extensions.size(); i++) {
    this->extensions[i]->handle_result(this->m_top_master_cache);
  }
}

void AbstractDuplicateDetector::post_handle_report(AbstractBugReport* report) {
  this->buckets->add_report(report);
}

AbstractDuplicateDetector::~AbstractDuplicateDetector() {
  //TODO clean the resource.
  delete this->buckets;
  this->buckets = NULL;
  delete this->topping_algorithm;
  this->topping_algorithm = NULL;
  delete this->report_list;
  this->report_list = NULL;

}
