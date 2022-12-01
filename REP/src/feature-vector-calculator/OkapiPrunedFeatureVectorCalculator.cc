/*
 * OkapiPrunedFeatureCalculator.cpp
 *
 *  Created on: Dec 7, 2010
 *      Author: Chengnian SUN
 */

#include "OkapiPrunedFeatureVectorCalculator.h"
#include <cstdio>
#include <cassert>

bool OkapiPrunedFeatureVectorCalculator::accept_feature_type(
    const enum SectionType::EnumSectionType,
    const enum SectionType::EnumSectionType doc_type,
    const enum IDFCollectionType::EnumIDFCollectionType idf_type) const {
  switch (idf_type) {
  case IDFCollectionType::IDF_BOTH: {
    return true;
  }
  case IDFCollectionType::IDF_DESC: {
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

OkapiPrunedFeatureVectorCalculator::OkapiPrunedFeatureVectorCalculator(
    const ReportBuckets& report_buckets, const unsigned textual_feature_count,
    const unsigned surface_feature_count) :
    OkapiFeatureVectorCalculator(report_buckets, textual_feature_count,
        surface_feature_count) {
}

OkapiPrunedFeatureVectorCalculator::OkapiPrunedFeatureVectorCalculator(
    const ReportBuckets& report_buckets) :
    OkapiFeatureVectorCalculator(report_buckets, 42, 0) {
}

OkapiPrunedFeatureVectorCalculator::~OkapiPrunedFeatureVectorCalculator() {
}
