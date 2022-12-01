/*
 * ICSE10PrunedFeatureCalculator.cpp
 *
 *  Created on: Dec 6, 2010
 *      Author: Chengnian SUN
 */

#include "ICSE10PrunedFeatureVectorCalculator.h"
#include <cstdio>
#include <iostream>
#include <cassert>
using namespace std;

bool ICSE10PrunedFeatureVectorCalculator::accept_feature_type(
    const enum SectionType::EnumSectionType,
    const enum SectionType::EnumSectionType doc_type,
    const enum IDFCollectionType::EnumIDFCollectionType idf_type) const {
  switch (idf_type) {
  case IDFCollectionType::IDF_BOTH: {
    return true;
  }
  case IDFCollectionType::IDF_DESC: {
    //		return doc_type == DESC_UNI || doc_type == DESC_BI || doc_type == DESC_TRI;
    return true;
  }
  case IDFCollectionType::IDF_SUMM: {
    return doc_type == SectionType::SUM_UNI || doc_type == SectionType::SUM_BI
        || doc_type == SectionType::SUM_TRI;
  }
  default:
    assert(false);
    fprintf(stderr, "ERROR: un-handled idf type %d\n", idf_type);
    exit(1);
  }
}

ICSE10PrunedFeatureVectorCalculator::ICSE10PrunedFeatureVectorCalculator(
    const ReportBuckets& report_buckets, const unsigned textual_feature_count,
    const unsigned surface_feature_count) :
    ICSE10FeatureVectorCalculator(report_buckets, textual_feature_count,
        surface_feature_count) {
}

ICSE10PrunedFeatureVectorCalculator::ICSE10PrunedFeatureVectorCalculator(
    const ReportBuckets& report_buckets) :
    ICSE10FeatureVectorCalculator(report_buckets, 42, 0) {

}

ICSE10PrunedFeatureVectorCalculator::~ICSE10PrunedFeatureVectorCalculator() {
}
