#pragma once

#ifndef _ICSE10_FEATURE_VECTOR_CALCULATOR_H__
#define _ICSE10_FEATURE_VECTOR_CALCULATOR_H__

#include "AbstractFeatureVectorCalculator.h"
#include "../tfidf/IDFCollectionType.h"
#include "../report-model/SectionType.h"

#include <vector>
using namespace std;

#define ICSE10_FEATURE_COUNT 54

class ICSE10FeatureVectorCalculator: public AbstractFeatureVectorCalculator {
private:
  void internal_create_textual_calculators(
      const vector<enum SectionType::EnumSectionType>& query_types,
      const vector<enum SectionType::EnumSectionType>& doc_types,
      vector<enum IDFCollectionType::EnumIDFCollectionType>& idf_types,
      vector<AbstractFeatureValueCalculator*>& result_collector) const;

protected:

  virtual AbstractFeatureValueCalculator* create_feature_value_calculator(
      const enum SectionType::EnumSectionType query_type,
      const enum SectionType::EnumSectionType doc_type,
      const enum IDFCollectionType::EnumIDFCollectionType idf_type) const;

  virtual vector<AbstractFeatureValueCalculator*> create_Textual_feature_vector_calculators() const;

  virtual vector<AbstractFeatureValueCalculator*> create_Surface_feature_vector_calculators() const;

  virtual bool accept_feature_type(
      const enum SectionType::EnumSectionType query_type,
      const enum SectionType::EnumSectionType doc_type,
      const enum IDFCollectionType::EnumIDFCollectionType idf_type) const;

  ICSE10FeatureVectorCalculator(const ReportBuckets& report_buckets,
      const unsigned textual_feature_count,
      const unsigned surface_feature_count);

public:

  ICSE10FeatureVectorCalculator(const ReportBuckets& report_buckets);

  virtual ~ICSE10FeatureVectorCalculator(void);

};

#endif /*_ICSE10_FEATURE_VECTOR_CALCULATOR_H__*/
