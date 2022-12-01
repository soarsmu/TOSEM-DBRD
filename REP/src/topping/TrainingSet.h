/*
 * TrainingSet.h
 *
 *  Created on: 2010-8-6
 *      Author: Chengnian Sun
 */

#ifndef TRAININGSET_H_
#define TRAININGSET_H_

#include <vector>
#include <utility>

#include "../libsvm/svm.h"

using namespace std;

class MasterBugReport;
class DuplicateBugReport;
class AbstractBugReport;
class ReportBuckets;

class TrainingInstance {
private:
  const DuplicateBugReport* query_report;
  const AbstractBugReport* base_report;
  int label;

public:
  inline const DuplicateBugReport* get_query_report() const {
    return this->query_report;
  }

  inline const AbstractBugReport* get_base_report() const {
    return this->base_report;
  }

  inline int get_label() const {
    return this->label;
  }

  TrainingInstance(const DuplicateBugReport* query_report,
      const AbstractBugReport* base_report, const int label) :
      query_report(query_report), base_report(base_report), label(label) {
  }

  ~TrainingInstance() {
  }
};

class TrainingSet {
private:

  const ReportBuckets* m_report_buckets;

  // the set to store initial queries
  vector<const DuplicateBugReport*> initial_query_set;

  // the set to store newly added queries.
  vector<const DuplicateBugReport*> new_query_set;

  /**
   * training instances., the first is the Query, the second is the Document.
   *
   * and Query and Document are duplicate
   */
  //vector<pair<DuplicateBugReport*, BugReport*> > relevant_training_instances;
  //vector<pair<DuplicateBugReport*, BugReport*> > irrelevant_training_instances;
  vector<TrainingInstance*> training_instances;

  // return the size of created relevant instances.
  unsigned int create_relevant_instances(
      const vector<const DuplicateBugReport*>& queries);

  void create_irrelevant_instances(
      const vector<const DuplicateBugReport*> & queries,
      const unsigned int number_to_create);

  inline unsigned int get_training_set_size() {
    //return this->relevant_training_instances.size() + this->irrelevant_training_instances.size();
    return this->training_instances.size();
  }

public:

  inline const vector<TrainingInstance*>& get_training_instances() const {
    return this->training_instances;
  }

  inline void set_report_buckets(const ReportBuckets& report_buckets) {
    this->m_report_buckets = &report_buckets;
  }

  inline void add_initial_query(const DuplicateBugReport* query) {
    this->initial_query_set.push_back(query);
  }

  inline void add_new_query(const DuplicateBugReport& query) {
    this->new_query_set.push_back(&query);
  }

  /**
   * this method call should be called after the initial query is constructed.
   */
  void populate_training_instances();

  TrainingSet(/*const unsigned random_seed*/);

  virtual ~TrainingSet();
};

#endif /* TRAININGSET_H_ */
