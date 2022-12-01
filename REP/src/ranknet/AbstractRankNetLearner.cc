/*
 * AbstractRankNetLearner.cc
 *
 *  Created on: Jan 10, 2011
 *      Author: Chengnian SUN
 */

#include <algorithm>
#include <cassert>
#include <cstdio>

#include "../detection-model/ReportBuckets.h"
#include "../report-model/MasterBugReport.h"
#include "../util/RandomUtility.h"

#include "../util/CmdOption.h"

#include "AbstractRankNetLearner.h"

using namespace std;

const static double TRAINING_MASTERS_RATIO = 1;

void AbstractRankNetLearner::build_training_pairs_for_a_pair_of_duplicates(
    const int bucket_id, const vector<MasterBugReport*>& masters,
    const AbstractBugReport* query_report,
    const AbstractBugReport* relevant_report) {
  const vector<MasterBugReport*>& random_masters =
      this->get_random_master_reports(masters,
//          this->m_default_model_parameter.get_count_of_irrelevant_reports_per_query(),
          this->m_count_of_irrelevant_reports_per_query, bucket_id);

  assert(
//      random_masters.size() == this->m_default_model_parameter.get_count_of_irrelevant_reports_per_query());
      random_masters.size() == this->m_count_of_irrelevant_reports_per_query);
  const unsigned count_random_masters = random_masters.size();
  for (unsigned random_index = 0; random_index < count_random_masters;
      random_index++) {
    // construct training pairs from the returned random IRRELEVANT masters.
    const MasterBugReport* random_master = random_masters[random_index];
    this->m_training_pairs.push_back(
        RankNetTrainingInstance(query_report, relevant_report, random_master));
  }
}

void AbstractRankNetLearner::build_training_pairs() {
#ifndef RANK_NET_TRACE_SIGN
  fprintf(this->get_log_file(), "INFO: RankNet: building training pairs...\n");

  fprintf(stdout, "INFO: RankNet: building training pairs...\n");
  fprintf(stdout, "INFO: RankNet: report repository size = %u\n",
      this->m_buckets.get_report_count());
  fprintf(stdout, "INFO: RankNet: report bucket count = %u\n",
      this->m_buckets.get_bucket_count());
  fprintf(stdout, "INFO: RankNet: master report count = %u\n",
      this->m_buckets.get_bucket_count());
#endif

  //	this->split_training_validating_sets();
  this->m_training_pairs.clear();

  const vector<MasterBugReport*>& masters =
      this->m_buckets.get_all_bucket_masters();

  const unsigned master_count = masters.size();
  //	const int first_validating_duplicate = this->get_first_validating_duplicate_id();

  unsigned query_size = 0;
  for (unsigned master_index = 0; master_index < master_count; master_index++) {
    // iterate over all masters.
    MasterBugReport* master = masters[master_index];
    if (master->has_no_duplicates()) {
      continue;
    }

    query_size++;
//    if (query_size >= this->m_default_model_parameter.get_max_query_count()) {
    if (query_size >= this->m_max_query_count) {
      break;
    }

    const int master_id = master->get_id();
    vector<const AbstractBugReport*> relevant_set;
    master->get_as_a_whole_bucket(relevant_set);

    const unsigned int relevant_set_size = relevant_set.size();

    for (unsigned query_index = 1; query_index < relevant_set_size;
        query_index++) {
      const AbstractBugReport* query_report = relevant_set[query_index];
      for (unsigned relevant_index = 0; relevant_index < query_index;
          relevant_index++) {
        const AbstractBugReport* relevant_report = relevant_set[relevant_index];

        this->build_training_pairs_for_a_pair_of_duplicates(master_id, masters,
            query_report, relevant_report);
        this->build_training_pairs_for_a_pair_of_duplicates(master_id, masters,
            relevant_report, query_report);
      }
    }
  }

  const unsigned expected_validation_size =
      static_cast<unsigned>(this->m_training_pairs.size()
          * (1 - TRAINING_MASTERS_RATIO));
  const
  unsigned expected_training_size = this->m_training_pairs.size()
      - expected_validation_size;

  for (vector<RankNetTrainingInstance>::iterator iter =
      this->m_training_pairs.begin() + expected_training_size;
      iter != this->m_training_pairs.end(); iter++) {
    this->m_validating_pairs.push_back(*iter);
  }
  this->m_training_pairs.erase(
      this->m_training_pairs.begin() + expected_training_size,
      this->m_training_pairs.end());

#ifndef RANK_NET_TRACE_SIGN
  fprintf(this->get_log_file(), "INFO: RankNet: built %u training pairs.\n",
      static_cast<unsigned>(this->m_training_pairs.size()));

  fprintf(this->get_log_file(), "INFO: RankNet: built %u validating pairs.\n",
      static_cast<unsigned>(this->m_validating_pairs.size()));

  fprintf(stdout, "INFO: RankNet: built %u training pairs.\n",
      static_cast<unsigned>(this->m_training_pairs.size()));

  fprintf(stdout, "INFO: RankNet: built %u validating pairs.\n",
      static_cast<unsigned>(this->m_validating_pairs.size()));
#endif

}


void AbstractRankNetLearner::build_training_pairs_from_file(string training_file_path) {
  #ifndef RANK_NET_TRACE_SIGN
  fprintf(this->get_log_file(), "INFO: RankNet: building training pairs...\n");

  fprintf(stdout, "INFO: RankNet: building training pairs...\n");
  fprintf(stdout, "INFO: RankNet: report repository size = %u\n",
      this->m_buckets.get_report_count());
  fprintf(stdout, "INFO: RankNet: report bucket count = %u\n",
      this->m_buckets.get_bucket_count());
  fprintf(stdout, "INFO: RankNet: master report count = %u\n",
      this->m_buckets.get_bucket_count());
#endif

  //	this->split_training_validating_sets();
  this->m_training_pairs.clear();

  const vector<MasterBugReport*>& masters =
      this->m_buckets.get_all_bucket_masters();

  const vector<AbstractBugReport*>&  all_reports = this->m_buckets.get_all_reports();

  FILE* file = fopen(training_file_path.c_str(), "r");
  if (!file) {
    string msg = "Cannot open training triplets file ";
    msg += training_file_path;
  }

  unsigned bug_id;
  unsigned relevant_id;
  unsigned irrelevant_id;

  int result;
  while ((result = fscanf(file, "%u,%u,%u", &bug_id, &relevant_id, &irrelevant_id)) >= 0) {
    if (!result) {
      continue;
    }
    if (result != 3) {
      printf("cannot reach here..\n");
    }

    AbstractBugReport* bug_report;
    AbstractBugReport* relevant;
    AbstractBugReport* irrelevant;
    const MasterBugReport* irrelevant_master;

    unsigned all_found = 0;
    for (unsigned idx=0; idx < all_reports.size(); idx++) {
      AbstractBugReport* cur_report = all_reports[idx];
      unsigned cur_id = cur_report->get_id();
      if (cur_id == bug_id) {
        bug_report = cur_report;
        all_found++;
      }
      if (cur_id == relevant_id) {
        relevant = cur_report;
        all_found++;
      }
      if (cur_id == irrelevant_id) {
        irrelevant = cur_report;
        irrelevant_master = irrelevant->get_master();
        all_found++;
      }
      if (all_found >= 3) {
        break;
      }
    }
    this->m_training_pairs.push_back(RankNetTrainingInstance(bug_report, relevant, irrelevant_master));
  }
  
  fclose(file);

#ifndef RANK_NET_TRACE_SIGN
  fprintf(this->get_log_file(), "INFO: RankNet: built %u training pairs.\n",
      static_cast<unsigned>(this->m_training_pairs.size()));

  fprintf(this->get_log_file(), "INFO: RankNet: built %u validating pairs.\n",
      static_cast<unsigned>(this->m_validating_pairs.size()));
#endif
}

static vector<int> excluding_ids_caches;

vector<MasterBugReport*> AbstractRankNetLearner::get_random_master_reports(
    const vector<MasterBugReport*>& masters, unsigned count_to_random,
    int excluding_master_id) {
  assert(count_to_random > 0);
  excluding_ids_caches.clear();
  excluding_ids_caches.push_back(excluding_master_id);

  vector<MasterBugReport*> result_vector;
  const unsigned master_count = masters.size() - 1;

  while (result_vector.size() < count_to_random) {
    MasterBugReport* random_master = masters[master_count
        * RandomUtility::get_default().random_double()];
    while (find(excluding_ids_caches.begin(), excluding_ids_caches.end(),
        random_master->get_id()) != excluding_ids_caches.end()) {
      random_master = masters[master_count
          * RandomUtility::get_default().random_double()];
    }
    excluding_ids_caches.push_back(random_master->get_id());
    result_vector.push_back(random_master);
  }
  return result_vector;
}

/*
 * RNC(Y) = log(1 + exp(Y)) where Y = y2 - y1, y2 and y1 are the similarity scores.
 *
 * RNC'(Y) = exp(Y) / (1 + exp(Y))
 *
 */
double AbstractRankNetLearner::compute_rnc_derivative_wrt_Y(
    const RankNetTrainingInstance& training_pair) const {
  const double similarity_difference = this->compute_similarity_difference(
      training_pair);
  const double exp_on_difference = exp(similarity_difference);
  const double result = exp_on_difference / (1 + exp_on_difference);
  assert(result == result);
  return result;
}

double AbstractRankNetLearner::compute_similarity_difference(
    const RankNetTrainingInstance& training_pair) const {
  const AbstractBugReport& query_report = training_pair.get_query();
  const AbstractBugReport& relevant_report =
      training_pair.get_relevant_report();

  const double relevance_similarity = this->compute_similarity(query_report,
      relevant_report);

  const AbstractBugReport& irrelevant_report =
      training_pair.get_irrelevant_report();
  const double irrelevance_similarity = this->compute_similarity(query_report,
      irrelevant_report);
  assert(relevance_similarity == relevance_similarity);
  assert(irrelevance_similarity == irrelevance_similarity);
  return irrelevance_similarity - relevance_similarity;
}

double AbstractRankNetLearner::compute_total_rnc_cost(
    const vector<RankNetTrainingInstance>& training_pairs) const {
  double result = 0;

  const unsigned number_of_training_pairs = training_pairs.size();
  for (unsigned i = 0; i < number_of_training_pairs; i++) {
    const RankNetTrainingInstance& training_pair = training_pairs[i];
    result += log(1 + exp(this->compute_similarity_difference(training_pair)));
  }
  return result;
}

void AbstractRankNetLearner::learn() {

  this->m_best_rnc_so_far = 999999;
  this->m_training_pairs.clear();
  this->m_validating_pairs.clear();
  // this->m_training_file = "/Users/ivanaclairineirsan/Documents/Research/DBRD/fast-dbrd-modified/dataset/eclipse/training_split_eclipse_triplets_random_1.txt";
  // string training_file = "/Users/ivanaclairineirsan/Documents/Research/DBRD/fast-dbrd-modified/dataset/eclipse/training_split_eclipse_triplets_random_1.txt";
  const CmdOption& option = CmdOption::get_instance();
  string training_file = option.get_train_file();
  
  if (training_file.size() > 0) {
    printf("Use %s as training data\n", training_file.c_str());
    this->build_training_pairs_from_file(training_file);
  } else {
    this->build_training_pairs();
  }
  
  const unsigned training_round_count = this->get_training_round_number();

  for (unsigned round = 1; round <= training_round_count; round++) {
#ifndef RANK_NET_TRACE_SIGN
    //		fprintf(stdout, "Round %u...\n", round);
#endif
    this->initialize_model_parameters(round);
    //	this->m_bm25f_parameter.initialize(2, true, 3, false, 0.5, false, 1, false, 1, true);

    double learning_rate = AbstractRankNetLearner::INITIAL_LEARNING_RATE;
    double previous_total_rnc_cost = 99999999;

    const unsigned number_of_training_pairs = this->m_training_pairs.size();

    for (unsigned epoch = 0; epoch < AbstractRankNetLearner::MAX_EPOCHES;
        epoch++) {
#ifndef RANK_NET_TRACE_SIGN
      //			fprintf(stdout, "Epoch %u...\n", epoch);
#endif
      for (unsigned i = 0; i < number_of_training_pairs; i++) {
        RankNetTrainingInstance& training_pair = this->m_training_pairs[i];
        this->before_tune_parameters_on_one_pair();
        this->tune_parameters_on_one_pair(training_pair, learning_rate);
        this->after_tune_parameters_on_one_pair();
      }

      // compute the total rnc cost
      const double new_rnc_cost = this->compute_total_rnc_cost(
          this->m_training_pairs);
      if (new_rnc_cost > previous_total_rnc_cost) {
        learning_rate /= 2;
      }
      previous_total_rnc_cost = new_rnc_cost;

      if (this->m_validating_pairs.size() > 0) {
        //			const double map = this->validate_wrt_MAP();
        const double validation_rnc = this->compute_total_rnc_cost(
            this->m_validating_pairs);
        if (validation_rnc < this->m_best_rnc_so_far) {
          this->m_best_rnc_so_far = validation_rnc;
          this->find_a_better_model();
        }
      } else {
        this->find_a_better_model();
      }

    } // end all epochs
  }

  this->learning_done();

}

AbstractRankNetLearner::AbstractRankNetLearner(FILE* log_file,
    const ReportBuckets& report_buckets,
    unsigned count_of_irrelevant_reports_per_query, unsigned max_query_count) :
    m_buckets(report_buckets), m_log_file(log_file), m_best_rnc_so_far(-99999), m_count_of_irrelevant_reports_per_query(
        count_of_irrelevant_reports_per_query), m_max_query_count(
        max_query_count) {
}

AbstractRankNetLearner::~AbstractRankNetLearner() {
}
