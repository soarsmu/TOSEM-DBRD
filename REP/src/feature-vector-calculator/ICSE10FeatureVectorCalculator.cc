#include "ICSE10FeatureVectorCalculator.h"

#include "../detection-model/ReportBuckets.h"
#include "../feature-value-calculator/IDFCrossFieldsFeatureValueCalculator.h"
#include <vector>
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <cstdio>
using namespace std;

AbstractFeatureValueCalculator* ICSE10FeatureVectorCalculator::create_feature_value_calculator(
    const enum SectionType::EnumSectionType query_type,
    const enum SectionType::EnumSectionType doc_type,
    const enum IDFCollectionType::EnumIDFCollectionType idf_type) const {
  return new IDFCrossFieldsFeatureValueCalculator(this->get_report_buckets(),
      query_type, doc_type, idf_type);
}

void ICSE10FeatureVectorCalculator::internal_create_textual_calculators(
    const vector<enum SectionType::EnumSectionType>& query_types,
    const vector<enum SectionType::EnumSectionType>& doc_types,
    vector<enum IDFCollectionType::EnumIDFCollectionType>& idf_types,
    vector<AbstractFeatureValueCalculator*>& result_collector) const {

  const unsigned size_of_query_types = query_types.size();
  const unsigned size_of_doc_types = doc_types.size();
  const unsigned size_of_idf_types = idf_types.size();

  for (unsigned i = 0; i < size_of_query_types; i++) {
    const enum SectionType::EnumSectionType query_type = query_types[i];

    for (unsigned j = 0; j < size_of_doc_types; j++) {
      const enum SectionType::EnumSectionType doc_type = doc_types[j];

      for (unsigned k = 0; k < size_of_idf_types; k++) {
        const enum IDFCollectionType::EnumIDFCollectionType idf_type =
            idf_types[k];
        if (this->accept_feature_type(query_type, doc_type, idf_type)) {
          result_collector.push_back(
              this->create_feature_value_calculator(query_type, doc_type,
                  idf_type));
        }
      }
    }
  }
}

vector<AbstractFeatureValueCalculator*> ICSE10FeatureVectorCalculator::create_Textual_feature_vector_calculators() const {

  vector<AbstractFeatureValueCalculator*> result;

  vector<enum SectionType::EnumSectionType> query_unigram_types;
  query_unigram_types.push_back(SectionType::SUM_UNI);
  query_unigram_types.push_back(SectionType::DESC_UNI);
  query_unigram_types.push_back(SectionType::ALL_UNI);

  vector<enum SectionType::EnumSectionType> doc_unigram_types;
  doc_unigram_types.push_back(SectionType::SUM_UNI);
  doc_unigram_types.push_back(SectionType::DESC_UNI);
  doc_unigram_types.push_back(SectionType::ALL_UNI);

  vector<enum IDFCollectionType::EnumIDFCollectionType> idf_types;
  idf_types.push_back(IDFCollectionType::IDF_BOTH);
  idf_types.push_back(IDFCollectionType::IDF_DESC);
  idf_types.push_back(IDFCollectionType::IDF_SUMM);

  this->internal_create_textual_calculators(query_unigram_types,
      doc_unigram_types, idf_types, result);
  /*
   *
   */
  vector<enum SectionType::EnumSectionType> query_bigram_types;
  query_bigram_types.push_back(SectionType::SUM_BI);
  query_bigram_types.push_back(SectionType::DESC_BI);
  query_bigram_types.push_back(SectionType::ALL_BI);

  vector<enum SectionType::EnumSectionType> doc_bigram_types;
  doc_bigram_types.push_back(SectionType::SUM_BI);
  doc_bigram_types.push_back(SectionType::DESC_BI);
  doc_bigram_types.push_back(SectionType::ALL_BI);

  this->internal_create_textual_calculators(query_bigram_types,
      doc_bigram_types, idf_types, result);

  return result;
}

vector<AbstractFeatureValueCalculator*> ICSE10FeatureVectorCalculator::create_Surface_feature_vector_calculators() const {
  // return an empty vector;
  return vector<AbstractFeatureValueCalculator*>();
}

bool ICSE10FeatureVectorCalculator::accept_feature_type(
    const enum SectionType::EnumSectionType,
    const enum SectionType::EnumSectionType,
    const enum IDFCollectionType::EnumIDFCollectionType) const {
  return true;
}

ICSE10FeatureVectorCalculator::ICSE10FeatureVectorCalculator(
    const ReportBuckets& report_buckets) :
    AbstractFeatureVectorCalculator(report_buckets, ICSE10_FEATURE_COUNT, 0) {

}
ICSE10FeatureVectorCalculator::ICSE10FeatureVectorCalculator(
    const ReportBuckets& report_buckets, const unsigned textual_feature_count,
    const unsigned surface_feature_count) :
    AbstractFeatureVectorCalculator(report_buckets, textual_feature_count,
        surface_feature_count) {
}

ICSE10FeatureVectorCalculator::~ICSE10FeatureVectorCalculator(void) {

}
