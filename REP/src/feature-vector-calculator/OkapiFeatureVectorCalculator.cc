#include <iostream>
#include "../okapi/OkapiWrapper.h"
#include "OkapiFeatureVectorCalculator.h"
#include "../feature-value-calculator/OkapiBM25CrossFieldFeatureValueCalculator.h"
using namespace std;

AbstractFeatureValueCalculator* OkapiFeatureVectorCalculator::create_feature_value_calculator(
    const enum SectionType::EnumSectionType query_type,
    const enum SectionType::EnumSectionType doc_type,
    const enum IDFCollectionType::EnumIDFCollectionType idf_type) const {
  return new OkapiBM25CrossFieldFeatureValueCalculator(
      this->get_report_buckets(), query_type, doc_type, idf_type);
}

OkapiFeatureVectorCalculator::OkapiFeatureVectorCalculator(
    const ReportBuckets& report_buckets, const unsigned textual_feature_count,
    const unsigned surface_feature_count) :
    ICSE10FeatureVectorCalculator(report_buckets, textual_feature_count,
        surface_feature_count) {
}

OkapiFeatureVectorCalculator::OkapiFeatureVectorCalculator(
    const ReportBuckets& report_buckets) :
    ICSE10FeatureVectorCalculator(report_buckets) {
}

OkapiFeatureVectorCalculator::~OkapiFeatureVectorCalculator(void) {
}
