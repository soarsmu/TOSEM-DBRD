#pragma once

#ifndef _ONLINE_RESULT_COLLECTOR_
#define _ONLINE_RESULT_COLLECTOR_

#include <climits>
#include <cstdio>
#include <ctime>
#include <vector>

#include "../detection-model/DetectionResultOfEachDuplicate.h"

#include "IDetectorExtension.h"

using namespace std;

class DuplicateBugReport;
class MasterBugReport;

class OnlineResultCollector: public IDetectorExtension {
private:

  vector<DetectionResultOfEachDuplicate> result_vector;

  unsigned int number_of_duplicates;

  // the starting time of the detection of the current duplicate report;
  time_t detection_start_time_per_duplicate;

  const AbstractBugReport* current_duplicate_report;

  const unsigned m_size_of_top_list;

  FILE* m_output_file;

  unsigned m_column;

public:

  vector<std::pair<double, int> > get_recall_result() const;

  double get_mean_average_precision() const;

  virtual void start_processing(const AbstractBugReport& report);

  virtual void set_comment(const char* comment);

  virtual void handle_result(const vector<const MasterBugReport*>& candidates);

  virtual void dispose();

  unsigned get_number_of_duplicates() const;

  OnlineResultCollector(unsigned int size_of_top_list, FILE* output_file =
      stdout);

  virtual ~OnlineResultCollector(void);
};

inline unsigned OnlineResultCollector::get_number_of_duplicates() const {
  return this->result_vector.size();
}

#endif /*_ONLINE_RESULT_COLLECTOR_*/
