/*
 * RecallAndTimeIntervalAnalysis.h
 *
 *  Created on: Apr 8, 2013
 *      Author: neo
 */

#ifndef _RECALL_AND_TIME_INTERVAL_ANALYSIS_H_
#define _RECALL_AND_TIME_INTERVAL_ANALYSIS_H_

#include <cassert>
#include <cstdio>
#include <vector>

#include "IDetectorExtension.h"

class DuplicateBugReport;

using std::vector;

class RecallAndTimeIntervalAnalysis: public IDetectorExtension {
private:

  const unsigned m_size_of_top_list;

  FILE* m_output_file;

  const AbstractBugReport* current_new_report;

public:

  virtual void start_processing(const AbstractBugReport& report);

  virtual void set_comment(const char* comment);

  virtual void handle_result(const vector<const MasterBugReport*>& candidates);

  virtual void dispose();

  RecallAndTimeIntervalAnalysis(unsigned size_of_top_list, FILE* output_file);

  virtual ~RecallAndTimeIntervalAnalysis();

};

#endif /* RECALLANDTIMEINTERVALANALYSIS_H_ */
