/*
 * Master2PartialBucket.cc
 *
 *  Created on: Mar 13, 2013
 *      Author: Chengnian Sun
 */

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <queue>

#include "../report-model/AbstractBugReport.h"

#include "Master2PartialBucket.h"

using std::less;
using std::queue;
using std::priority_queue;

struct ReportSimilarityLess {
  bool operator()(const AbstractBugReport* x,
      const AbstractBugReport* y) const {
    return x->get_similarity_info()->get_similarity()
        < y->get_similarity_info()->get_similarity();
  }
};

void Master2PartialBucket::retrieval_done(const AbstractBugReport&) {
  if (this->m_top_cosine_number == 0) {
    return;
  }
  std::priority_queue<const AbstractBugReport*,
      vector<const AbstractBugReport*>, ReportSimilarityLess> q;
  for (Master2PartialBucket::MapIterator iter = this->m_buckets.begin(), end =
      this->m_buckets.end(); iter != end; ++iter) {
    const Master2PartialBucket::PartialBucket* bucket = iter->second;
    assert(bucket->size());
    for (PartialBucket::const_iterator biter = bucket->begin(), bend =
        bucket->end(); biter != bend; ++biter) {
      const ReportWithSimilarity& element = *biter;
      const AbstractBugReport* report = element.get_report();
      SimilarityInfo& sim_info = *report->get_similarity_info();
      sim_info.set_similarity(element.get_similarity(), 0);
      q.push(report);
    }
  }
  assert(this->m_accepted.empty());
#ifndef NDEBUG
  double last_similarity = -99999;
#endif

  for (unsigned i = 0; i < this->m_top_cosine_number && q.size(); ++i) {

#ifndef NDEBUG
    const double current_simi =
        q.top()->get_similarity_info()->get_similarity();
    assert(last_similarity < 0 || last_similarity >= current_simi);
    last_similarity = q.top()->get_similarity_info()->get_similarity();
#endif
    this->m_accepted.insert(q.top()->get_id());
    q.pop();
  }
}

Master2PartialBucket::PartialBucket* Master2PartialBucket::get_or_create_bucket() {
  PartialBucket* bucket;
  if (this->m_unused.size()) {
    bucket = this->m_unused.back();
    this->m_unused.pop_back();
  } else {
    bucket = new PartialBucket();
    bucket->reserve(20);
  }
  this->m_used.push_back(bucket);
  return bucket;
}

void Master2PartialBucket::add(Master2PartialBucket::MasterID master_id,
    const AbstractBugReport* report, const float similarity_delta) {
  PartialBucket* bucket = this->m_buckets[master_id];
  if (bucket == NULL) {

    bucket = this->get_or_create_bucket();
    this->m_buckets[master_id] = bucket;
    assert(bucket->empty());
    bucket->push_back(ReportWithSimilarity(report, similarity_delta));
    ++this->m_number_of_reports;

  } else {
    bool found = false;
    const int report_id = report->get_id();
    for (PartialBucket::iterator iter = bucket->begin(), end = bucket->end();
        iter != end; ++iter) {
      if ((*iter).get_report()->get_id() == report_id) {
        found = true;
//        iter->m_similarity++;
        iter->increase_similarity(similarity_delta);
        break;
      }
    }
    if (!found) {
      bucket->push_back(ReportWithSimilarity(report, similarity_delta));
      ++this->m_number_of_reports;
    }
  }
}

void Master2PartialBucket::clear() {
  this->m_accepted.clear();
  this->m_buckets.clear();
  for (vector<PartialBucket*>::const_iterator iter = this->m_used.begin(), end =
      this->m_used.end(); iter != end; ++iter) {
    PartialBucket* bucket = *iter;
    bucket->clear();
    this->m_unused.push_back(bucket);
  }
  this->m_used.clear();

  this->m_number_of_reports = 0;
  assert(this->m_used.empty());
  assert(this->m_buckets.empty());
}

Master2PartialBucket::Master2PartialBucket(unsigned top_cosine_number) :
    m_top_cosine_number(top_cosine_number) {
  this->m_number_of_reports = 0;
}

Master2PartialBucket::~Master2PartialBucket() {
  for (vector<PartialBucket*>::const_iterator iter = this->m_used.begin(), end =
      this->m_used.end(); iter != end; ++iter) {
    delete *iter;
  }
  for (vector<PartialBucket*>::const_iterator iter = this->m_unused.begin(),
      end = this->m_unused.end(); iter != end; ++iter) {
    delete *iter;
  }
  this->m_used.clear();
  this->m_unused.clear();
}

