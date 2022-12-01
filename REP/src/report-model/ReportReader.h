/*
 * ReportReader.h
 *
 *  Created on: 2010-7-29
 *      Author: Chengnian Sun.
 */

#ifndef _REPORT_READER_H_
#define _REPORT_READER_H_

#include <string>
#include <vector>

class AbstractBugReport;

using namespace std;

class ReportReader {
private:

  int max_term_id;

  void read_reports_from_file(const string& file_path,
      const string& timestamp_file,
      vector<AbstractBugReport*>* report_collector);

public:

  int get_max_term_id() const;

  ReportReader(const string& file_path, const string& timestamp_file,
      vector<AbstractBugReport*>* report_collector);

  ~ReportReader();

};

#endif /* REPORTREADER_H_ */
