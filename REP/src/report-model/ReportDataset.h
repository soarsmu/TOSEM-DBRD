/*
 * ReportDataset.h
 *
 *  Created on: Jan 14, 2011
 *      Author: Chengnian SUN
 */

#ifndef REPORTDATASET_H_
#define REPORTDATASET_H_

#include <string>
#include <vector>
class AbstractBugReport;

using namespace std;

class ReportDataset {
private:

  vector<AbstractBugReport*>* m_all_reports_in_dataset;

  int m_max_term_id;

public:

  int get_max_term_id() const;

  void get_copy(vector<AbstractBugReport*>& copy_collector) const;

  explicit ReportDataset(const string& report_dataset_path,
      const string& timestamp_file);

  virtual ~ReportDataset();
};

inline int ReportDataset::get_max_term_id() const {
  return this->m_max_term_id;
}

#endif /* REPORTDATASET_H_ */
