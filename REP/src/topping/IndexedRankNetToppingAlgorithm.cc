/*
 * RankNetToppingAlgorithmWithIndex.cc
 *
 *  Created on: Apr 4, 2013
 *      Author: Chengnian Sun
 */

#include <boost/unordered_set.hpp>
#include <cassert>
#include <climits>
#include <cstdio>
#include <iostream>
#include <vector>

#include "../detection-model/ReportBuckets.h"
#include "../index/InvertedIndex.h"
#include "../index/Master2PartialBucket.h"
#include "../index/Postings.h"
#include "../okapi/OkapiWrapper.h"
#include "../ranknet/AbstractRankNetLearner.h"
#include "../report-model/Term.h"
#include "../util/CmdOption.h"
#include "../util/MacroUtility.h"
#include "IndexedRankNetToppingAlgorithm.h"

using boost::unordered_set;
using namespace std;

IndexedRankNetToppingAlgorithm::IndexedRankNetToppingAlgorithm(FILE* log_file,
    const ReportBuckets& buckets, const DefaultREPParameter& parameter) :
    RankNetToppingAlgorithm(log_file, buckets, parameter), m_idf_threshold(
        CmdOption::get_instance().get_indexing_idf_threshold()), master_2_partial_bucket(
        new Master2PartialBucket(UINT_MAX)) {
  this->m_index = new InvertedIndex(buckets.get_max_term_id(),
      buckets.get_both_idf_collection());
  this->m_size_of_indexed_reports = 0;
}

IndexedRankNetToppingAlgorithm::~IndexedRankNetToppingAlgorithm() {
  delete this->m_index;
  this->m_index = NULL;
  delete this->master_2_partial_bucket;
  this->master_2_partial_bucket = NULL;
}

void IndexedRankNetToppingAlgorithm::build_index() {
  vector<PreciseTerm> normalized_terms_cache;
  normalized_terms_cache.reserve(256);

  const ReportBuckets& buckets = this->get_buckets();
  InvertedIndex& index = *this->m_index;
  const OkapiWrapper::AverageLengthInfo unigram_average_length_info(
      buckets.get_average_length_of_summary_Unigram_section(),
      buckets.get_average_length_of_description_Unigram_section());

  const vector<AbstractBugReport*>& all_reports = buckets.get_all_reports();
  const unsigned number_of_reports = all_reports.size();
  for (; this->m_size_of_indexed_reports < number_of_reports;
      ++this->m_size_of_indexed_reports) {
    const AbstractBugReport* report =
        all_reports[this->m_size_of_indexed_reports];

    assert(normalized_terms_cache.empty());

    const StructuredSection& unigram_sections =
        report->get_Unigram_structured_section();

    const OkapiWrapper::LengthInfo length_info(
        unigram_sections.get_summary_length(),
        unigram_sections.get_description_length());

    const vector<StructuredTerm>& terms = unigram_sections.get_terms();
    for (vector<StructuredTerm>::const_iterator iter = terms.begin(), end =
        terms.end(); iter != end; ++iter) {
      const StructuredTerm& term = *iter;
      const float term_weight = this->get_learner()->compute_term_weight_in_doc(
          term.get_summary_tf(), term.get_description_tf(), length_info,
          unigram_average_length_info);
      normalized_terms_cache.push_back(
          PreciseTerm(term.get_tid(), term_weight));
    }
    index.add_report(normalized_terms_cache, *report);
    normalized_terms_cache.clear();
  }
}

//static void retrieve_from_index(Master2PartialBucket& master_2_partial_bucket,
//    InvertedIndex* index, const AbstractRankNetLearner* learner,
//    IDFCollection* idf_collection, const AbstractBugReport& query_report,
//    const float idf_threshold) {
//  assert(master_2_partial_bucket.empty());
//
//  const vector<StructuredTerm> & query_terms =
//      query_report.get_Unigram_structured_section().get_terms();
//  for (vector<StructuredTerm>::const_iterator iter = query_terms.begin(), end =
//      query_terms.end(); iter != end; ++iter) {
//    const StructuredTerm& query_term = *iter;
//
//    const float idf = idf_collection->get_idf(query_term.get_tid());
//
//    if (idf < idf_threshold)
//      continue;
//
//    const float query_weight = learner->compute_term_weight_in_query(
//        query_term.get_summary_tf(), query_term.get_description_tf());
//    const float idf_and_query_weight = idf * query_weight;
//    const Postings& postings = *index->get_postings(query_term.get_tid());
//    const vector<TermInReport>& reports = postings.get_terms();
//
//    for (vector<TermInReport>::const_iterator riter = reports.begin(), rend =
//        reports.end(); riter != rend; ++riter) {
//      const TermInReport& term_in_report = *riter;
//      const AbstractBugReport* bug_report = term_in_report.get_report();
//      const MasterBugReport* master = bug_report->get_master();
//
//      master_2_partial_bucket.add(master->get_id(), bug_report,
//          term_in_report.get_normalized_tf() * idf_and_query_weight);
//    }
//  }
//}

static void retrieve_from_index(Master2PartialBucket& master_2_partial_bucket,
    InvertedIndex* index, const AbstractRankNetLearner* learner,
    IDFCollection* idf_collection, const AbstractBugReport& query_report,
    const float idf_threshold) {
  assert(master_2_partial_bucket.empty());

  const vector<StructuredTerm> & query_terms =
      query_report.get_Unigram_structured_section().get_terms();
  for (vector<StructuredTerm>::const_iterator iter = query_terms.begin(), end =
      query_terms.end(); iter != end; ++iter) {
    const StructuredTerm& query_term = *iter;

    const float idf = idf_collection->get_idf(query_term.get_tid());

    if (idf < idf_threshold)
      continue;

    const float query_weight = learner->compute_term_weight_in_query(
        query_term.get_summary_tf(), query_term.get_description_tf());
    const float idf_and_query_weight = idf * query_weight;
    const Postings& postings = *index->get_postings(query_term.get_tid());
    const vector<TermInReport>& reports = postings.get_terms();

    for (vector<TermInReport>::const_iterator riter = reports.begin(), rend =
        reports.end(); riter != rend; ++riter) {
      const TermInReport& term_in_report = *riter;
      const AbstractBugReport* bug_report = term_in_report.get_report();
      const MasterBugReport* master = bug_report->get_master();

      master_2_partial_bucket.add(master->get_id(), bug_report,
          term_in_report.get_normalized_tf() * idf_and_query_weight);
    }
  }
}

void IndexedRankNetToppingAlgorithm::internal_get_top(
    const AbstractBugReport& query_report, MasterReportPriorityQueue& queue) {
  assert(queue.empty());
  const AbstractRankNetLearner* learner = this->get_learner();
  this->master_2_partial_bucket->clear();

  retrieve_from_index(*master_2_partial_bucket, this->m_index, learner,
      this->get_buckets().get_both_idf_collection(), query_report,
      this->m_idf_threshold);

  printf("%u out of %u\n", this->master_2_partial_bucket->number_of_reports(),
      this->get_buckets().get_all_reports().size());

  for (Master2PartialBucket::MapIterator iter =
      master_2_partial_bucket->begin(), end = master_2_partial_bucket->end();
      iter != end; ++iter) {
    const Master2PartialBucket::PartialBucket* bucket = iter->second;
    assert(bucket->size());
    Master2PartialBucket::PartialBucket::const_iterator biter = bucket->begin();
    const MasterBugReport* master = (*biter).get_report()->get_master();

    double max = -9999;
    int similar_report = -1;
    for (Master2PartialBucket::PartialBucket::const_iterator bend =
        bucket->end(); biter != bend; ++biter) {
      const Master2PartialBucket::ReportWithSimilarity& element = *biter;
      const AbstractBugReport& report = *element.get_report();
      const int report_id = report.get_id();
//      const double report_sim = this->compute_similarity(query_report, *report);
      const float text_similarity = learner->weigh_textual_similarity(
          element.get_similarity());
      const float report_sim = learner->compute_similarity_with_textual_sim(
          text_similarity, query_report, report);
      if (max < report_sim) {
        max = report_sim;
        similar_report = report_id;
      }
    }
    master->get_similarity_info()->set_similarity(max, similar_report);
    queue.add(master);
  }

}

void IndexedRankNetToppingAlgorithm::before_get_top() {
  RankNetToppingAlgorithm::before_get_top();
  build_index();
}

