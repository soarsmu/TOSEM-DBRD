/*
 * ReportBuckets.cpp
 *
 *  Created on: 2010-7-29
 *      Author: Chengnian Sun.
 */

#include <cassert>
#include <iostream>
#include <stdio.h>

#include "../report-model/AbstractBugReport.h"
#include "../report-model/DuplicateBugReport.h"
#include "../report-model/MasterBugReport.h"
#include "../tfidf/IDFCollection.h"
#include "../util/MacroUtility.h"

#include "CommonIndexingPolicy.h"
#include "NoIndexingPolicy.h"
#include "ReportBuckets.h"

using namespace std;

void ReportBuckets::reset_similarity() const {
  for (vector<AbstractBugReport*>::const_iterator iter =
      this->m_all_reports.begin(), end = this->m_all_reports.end(); iter != end;
      ++iter) {
    const AbstractBugReport* report = *iter;
    report->get_similarity_info()->reset();
  }
}

MasterBugReport* ReportBuckets::get_master(int id) const {
  assert(m_master_index.count(id));
  MasterBugReport* master = m_master_index.find(id)->second;
  assert(master != NULL);
  return master;
}

void ReportBuckets::add_report_to_idf_collections(AbstractBugReport *report) {

  this->total_idf_collection->add_one_report(
      report->get_all_unigrams().get_terms(),
      report->get_all_bigrams().get_terms());

  this->summary_idf_collection->add_one_report(
      report->get_summary_unigrams().get_terms(),
      report->get_summary_bigrams().get_terms());

  this->description_idf_collection->add_one_report(
      report->get_description_unigrams().get_terms(),
      report->get_description_bigrams().get_terms());

  assert(
      this->get_report_count() == this->total_idf_collection->number_of_documents());
  assert(
      this->get_report_count() == this->summary_idf_collection->number_of_documents());
  assert(
      this->get_report_count() == this->description_idf_collection->number_of_documents());
}

void ReportBuckets::update_average_document_lengths(
    const unsigned int number_of_reports, AbstractBugReport* report) {
  const unsigned int old_number_of_reports = number_of_reports - 1;
  assert(old_number_of_reports < number_of_reports);

  this->average_length_of_summary_unigrams =
      (this->average_length_of_summary_unigrams * old_number_of_reports
          + report->get_summary_unigrams().get_length()) / number_of_reports;

  this->average_length_of_summary_bigrams =
      (this->average_length_of_summary_bigrams * old_number_of_reports
          + report->get_summary_bigrams().get_length()) / number_of_reports;

  this->average_length_of_summary_trigrams =
      (this->average_length_of_summary_trigrams * old_number_of_reports
          + report->get_summary_trigrams().get_length()) / number_of_reports;

  this->average_length_of_description_unigrams =
      (this->average_length_of_description_unigrams * old_number_of_reports
          + report->get_description_unigrams().get_length())
          / number_of_reports;

  this->average_length_of_description_bigrams =
      (this->average_length_of_description_bigrams * old_number_of_reports
          + report->get_description_bigrams().get_length()) / number_of_reports;

  this->average_length_of_description_trigrams =
      (this->average_length_of_description_trigrams * old_number_of_reports
          + report->get_description_trigrams().get_length())
          / number_of_reports;

  this->average_length_of_all_unigrams = (this->average_length_of_all_unigrams
      * old_number_of_reports + report->get_all_unigrams().get_length())
      / number_of_reports;

  this->average_length_of_all_bigrams = (this->average_length_of_all_bigrams
      * old_number_of_reports + report->get_all_bigrams().get_length())
      / number_of_reports;

  this->average_length_of_all_trigrams = (this->average_length_of_all_trigrams
      * old_number_of_reports + report->get_all_trigrams().get_length())
      / number_of_reports;
}

double ReportBuckets::get_average_length_of_sections(
    enum SectionType::EnumSectionType type) const {
  switch (type) {
  case SectionType::SUM_UNI:
    return this->average_length_of_summary_unigrams;
  case SectionType::SUM_BI:
    return this->average_length_of_summary_bigrams;
  case SectionType::SUM_TRI:
    return this->average_length_of_summary_trigrams;

  case SectionType::DESC_UNI:
    return this->average_length_of_description_unigrams;
  case SectionType::DESC_BI:
    return this->average_length_of_description_bigrams;
  case SectionType::DESC_TRI:
    return this->average_length_of_description_trigrams;

  case SectionType::ALL_UNI:
    return this->average_length_of_all_unigrams;
  case SectionType::ALL_BI:
    return this->average_length_of_all_bigrams;
  case SectionType::ALL_TRI:
    return this->average_length_of_all_trigrams;
  default:
    assert(false);
    return 0;
  }
}

void ReportBuckets::add_report(AbstractBugReport* report) {
  this->m_all_reports.push_back(report);

  this->m_indexing_policy->update_index(*report);

  if (!(report->is_duplicate())) {
    MasterBugReport* master = static_cast<MasterBugReport*>(report);
    this->m_master_index[report->get_id()] = master;
    this->m_masters.push_back(master);
  } else {
    MasterBugReport* master = this->get_master(report->get_duplicate_id());
    assert(master != NULL);
    DuplicateBugReport* duplicate = static_cast<DuplicateBugReport*>(report);
    master->add_duplicate(duplicate);
    duplicate->set_master(master);

    //		this->m_duplicate_reports.push_back(duplicate);
  }

  this->add_report_to_idf_collections(report);
  this->update_average_document_lengths(this->m_all_reports.size(), report);
}

static AbstractIndexingPolicy* create_indexing_policy(int max_term_id,
    IndexingType::EnumIndexingType indexing_type,
    IDFCollection* idf_collection) {
  switch (indexing_type) {
  case IndexingType::NO_INDEXING:
    return new NoIndexingPolicy();
  default:
    return new CommonIndexingPolicy(max_term_id, indexing_type, idf_collection);
  }
}

IndexingType::EnumIndexingType ReportBuckets::get_report_candidates(
    const AbstractBugReport& query_report,
    Master2PartialBucket& candidate_collector) const {
  assert(this->m_indexing_policy);
  return this->m_indexing_policy->get_report_candidates(query_report,
      candidate_collector);
}

ReportBuckets::ReportBuckets(int max_term_id,
    IndexingType::EnumIndexingType indexing_type) :
    m_max_term_id(max_term_id) {
  //this->report_count = 0;
  this->total_idf_collection = new IDFCollection(max_term_id);
  this->summary_idf_collection = new IDFCollection(max_term_id);
  this->description_idf_collection = new IDFCollection(max_term_id);

  this->average_length_of_summary_unigrams = 0;
  this->average_length_of_summary_bigrams = 0;
  this->average_length_of_summary_trigrams = 0;

  this->average_length_of_description_unigrams = 0;
  this->average_length_of_description_bigrams = 0;
  this->average_length_of_description_trigrams = 0;

  this->average_length_of_all_unigrams = 0;
  this->average_length_of_all_bigrams = 0;
  this->average_length_of_all_trigrams = 0;

  assert(this->total_idf_collection);
  assert(this->summary_idf_collection);
  assert(this->description_idf_collection);
  m_indexing_policy = create_indexing_policy(max_term_id, indexing_type,
      this->total_idf_collection);
}

ReportBuckets::~ReportBuckets() {
  delete this->total_idf_collection;
  this->total_idf_collection = NULL;
  delete this->summary_idf_collection;
  this->summary_idf_collection = NULL;
  delete this->description_idf_collection;
  this->description_idf_collection = NULL;

  delete this->m_indexing_policy;

}

IDFCollection* ReportBuckets::get_idf_collection(
    enum IDFCollectionType::EnumIDFCollectionType type) const {
  switch (type) {
  case IDFCollectionType::IDF_SUMM:
    return this->summary_idf_collection;
  case IDFCollectionType::IDF_DESC:
    return this->description_idf_collection;
  case IDFCollectionType::IDF_BOTH:
    return this->total_idf_collection;
  default:
//    assert(false);
    UNREACHABLE("Unhandled type...");
    return this->summary_idf_collection;
  }
}

