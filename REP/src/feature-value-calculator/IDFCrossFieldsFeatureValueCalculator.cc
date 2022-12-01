/*
 * IDFCrossFieldsFeatureValueCalculator.cc
 *
 *  Created on: Jan 6, 2011
 *      Author: Chengnian SUN
 */

#include "IDFCrossFieldsFeatureValueCalculator.h"

double IDFCrossFieldsFeatureValueCalculator::weigh_term(const double idf,
    const double, const double, const double, const double) const {
  return idf;
}

IDFCrossFieldsFeatureValueCalculator::IDFCrossFieldsFeatureValueCalculator(
    const ReportBuckets& report_buckets,
    const SectionType::EnumSectionType query_type,
    const SectionType::EnumSectionType doc_type,
    const IDFCollectionType::EnumIDFCollectionType idf_type) :
    CrossFieldsFeatureValueCalculator(report_buckets, query_type, doc_type,
        idf_type) {
}

IDFCrossFieldsFeatureValueCalculator::~IDFCrossFieldsFeatureValueCalculator() {
}
