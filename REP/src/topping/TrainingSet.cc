/*
 * TrainingSet.cpp
 *
 *  Created on: 2010-8-6
 *      Author: Chengnian Sun
 */

#include "TrainingSet.h"
#include <cassert>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <set>
using namespace std;

#include "../util/MacroUtility.h"
#include "../util/RandomUtility.h"

#include "../report-model/MasterBugReport.h"
#include "../report-model/DuplicateBugReport.h"
#include "../detection-model/ReportBuckets.h"

#define TRAINING_SET_UPPER_BOUND 6000

#define RANDOM_SEED 1282488393

unsigned int TrainingSet::create_relevant_instances(
    const vector<const DuplicateBugReport*>& queries) {
  unsigned int number_of_created_relevant_instances = 0;

  const unsigned int query_count = queries.size();
  for (unsigned int query_index = 0; query_index < query_count; query_index++) {
    const DuplicateBugReport* query = queries[query_index];
    const MasterBugReport* master = query->get_master();

    number_of_created_relevant_instances++;
    this->training_instances.push_back(new TrainingInstance(query, master, 1));

    const vector<const DuplicateBugReport*>& duplicates =
        master->get_duplicates();
    const unsigned int dup_count = duplicates.size();
    for (unsigned int dup_index = 0; dup_index < dup_count; dup_index++) {
      number_of_created_relevant_instances++;
      this->training_instances.push_back(
          new TrainingInstance(query, duplicates[dup_index], 1));
    }
  }

  return number_of_created_relevant_instances;
}

void TrainingSet::create_irrelevant_instances(
    const vector<const DuplicateBugReport*> & queries,
    const unsigned int number_to_create) {
  if (number_to_create <= 0) {
    return;
  }

  const unsigned int query_count = queries.size();
  const unsigned int each = number_to_create / query_count + 1;

  const int bucket_size = this->m_report_buckets->get_bucket_count();
  set<int> created;

  for (unsigned int query_index = 0; query_index < query_count; query_index++) {
    const DuplicateBugReport* query = queries[query_index];
    for (unsigned i = 0; i < each; i++) {
      MasterBugReport* master = NULL;
      while (master == NULL) {
        master = this->m_report_buckets->get_bucket_master(
            static_cast<int>(RandomUtility::get_default().random_double()
                * (bucket_size - 1)));
        if ((master->get_id() == query->get_duplicate_id())
            || (created.find(master->get_id()) != created.end())) {
          master = NULL;
        } else {
          created.insert(master->get_id());
        }
      }
      this->training_instances.push_back(
          new TrainingInstance(query, master, 0));
    }
  }

}

void TrainingSet::populate_training_instances() {
  if (this->initial_query_set.size() != 0) {
    const unsigned int number_of_created_relevant_instances =
        this->create_relevant_instances(this->initial_query_set);
    this->create_irrelevant_instances(this->initial_query_set,
        number_of_created_relevant_instances);
    this->initial_query_set.clear();
  } else {
    if (this->get_training_set_size() < TRAINING_SET_UPPER_BOUND) {
      const unsigned int number_of_created_relevant_instances =
          this->create_relevant_instances(this->new_query_set);
      this->create_irrelevant_instances(this->new_query_set,
          number_of_created_relevant_instances);
    }
    this->new_query_set.clear();
  }
}

TrainingSet::TrainingSet() :
    m_report_buckets(NULL) {
}

TrainingSet::~TrainingSet() {
  for (int i = this->training_instances.size() - 1; i > -1; i--) {
    delete this->training_instances[i];
    this->training_instances[i] = NULL;
  }
}
