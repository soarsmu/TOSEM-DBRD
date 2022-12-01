/*
 * ReportBuckets.h
 *
 *  Created on: 2010-7-29
 *      Author: Chengnian Sun.
 */

#ifndef REPORTBUCKETS_H_
#define REPORTBUCKETS_H_

#include <boost/unordered_map.hpp>
#include <vector>
#include <iostream>
#include <cassert>

#include "../report-model/SectionType.h"
#include "../tfidf/IDFCollectionType.h"

#include "IndexingType.h"

using namespace std;

class AbstractIndexingPolicy;
class AbstractBugReport;
class IDFCollection;
class Master2PartialBucket;
class MasterBugReport;

/**
 * A ReportBuckets is actually a representation of bug repository.
 *
 *
 */
class ReportBuckets {

private:

  const AbstractIndexingPolicy* m_indexing_policy;

  boost::unordered_map<int, MasterBugReport*> m_master_index;

  vector<MasterBugReport*> m_masters;

  const unsigned m_max_term_id;

  // store all the reports in the repository
  vector<AbstractBugReport*> m_all_reports;

  /**
   * idf collections.
   */
  IDFCollection *total_idf_collection;
  IDFCollection *summary_idf_collection;
  IDFCollection *description_idf_collection;

  double average_length_of_summary_unigrams;
  double average_length_of_summary_bigrams;
  double average_length_of_summary_trigrams;

  double average_length_of_description_unigrams;
  double average_length_of_description_bigrams;
  double average_length_of_description_trigrams;

  double average_length_of_all_unigrams;
  double average_length_of_all_bigrams;
  double average_length_of_all_trigrams;

  void add_report_to_idf_collections(AbstractBugReport *report);

  void update_average_document_lengths(unsigned int number_of_reports,
      AbstractBugReport* report);

public:

  void reset_similarity() const;

  unsigned get_max_term_id() const;

  double get_average_length_of_summary_Unigram_section() const;

  double get_average_length_of_summary_Bigram_section() const;

  double get_average_length_of_description_Unigram_section() const;

  double get_average_length_of_description_Bigram_section() const;

  double get_average_length_of_sections(
      enum SectionType::EnumSectionType type) const;

  IDFCollection* get_both_idf_collection() const;

  IDFCollection* get_idf_collection(
      enum IDFCollectionType::EnumIDFCollectionType type) const;

  unsigned get_report_count() const;

  const vector<AbstractBugReport*>& get_all_reports() const;

  const vector<MasterBugReport*>& get_all_bucket_masters() const;

  MasterBugReport* get_master(int id) const;

  MasterBugReport* get_bucket_master(const int bucket_id) const;

  unsigned get_bucket_count() const;

  unsigned get_duplicates_count() const;

  void add_report(AbstractBugReport* report);

  IndexingType::EnumIndexingType get_report_candidates(
      const AbstractBugReport& query_report,
      Master2PartialBucket& candicate_collector) const;

  ReportBuckets(int max_term_id, IndexingType::EnumIndexingType indexing_type);

  ~ReportBuckets();
};

inline IDFCollection* ReportBuckets::get_both_idf_collection() const {
  return this->total_idf_collection;
}

inline unsigned ReportBuckets::get_max_term_id() const {
  return this->m_max_term_id;
}

inline unsigned ReportBuckets::get_bucket_count() const {
  return this->m_masters.size();
}

inline unsigned ReportBuckets::get_duplicates_count() const {
  return this->get_report_count() - this->get_bucket_count();
}

inline unsigned ReportBuckets::get_report_count() const {
  return this->m_all_reports.size();
}

inline const vector<AbstractBugReport*>&
ReportBuckets::get_all_reports() const {
  return this->m_all_reports;
}

inline const vector<MasterBugReport*>&
ReportBuckets::get_all_bucket_masters() const {
  return this->m_masters;
}

inline double
ReportBuckets::get_average_length_of_summary_Unigram_section() const {
  return this->average_length_of_summary_unigrams;
}

inline double
ReportBuckets::get_average_length_of_summary_Bigram_section() const {
  return this->average_length_of_summary_bigrams;
}

inline double
ReportBuckets::get_average_length_of_description_Unigram_section() const {
  return this->average_length_of_description_unigrams;
}

inline double
ReportBuckets::get_average_length_of_description_Bigram_section() const {
  return this->average_length_of_description_bigrams;
}

inline MasterBugReport* ReportBuckets::get_bucket_master(
    const int bucket_id) const {
  assert(bucket_id < static_cast<int>(this->m_masters.size()));
  return this->m_masters[bucket_id];
}

#endif /* REPORTBUCKETS_H_ */
