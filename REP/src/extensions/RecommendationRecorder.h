/*
 * RecommendationRecorder.h
 *
 *  Created on: Aug 30, 2011
 *      Author: Chengnian Sun
 */

#ifndef RECOMMENDATIONRECORDER_H_
#define RECOMMENDATIONRECORDER_H_

#include <cassert>
#include <cstdio>
#include <vector>

#include "IDetectorExtension.h"

class DuplicateBugReport;

using std::vector;

class RecommendationRecorder: public IDetectorExtension {
private:

  const unsigned m_size_of_top_list;

  FILE* m_output_file;

  // the current duplicate report;
  const AbstractBugReport* current_new_report;

public:

  virtual void start_processing(const AbstractBugReport& report);

  virtual void set_comment(const char* comment);

  virtual void handle_result(const vector<const MasterBugReport*>& candidates);

  virtual void dispose();

  RecommendationRecorder(unsigned size_of_top_list, FILE* output_file);

  virtual ~RecommendationRecorder();

};

#endif /* RECOMMENDATIONRECORDER_H_ */
