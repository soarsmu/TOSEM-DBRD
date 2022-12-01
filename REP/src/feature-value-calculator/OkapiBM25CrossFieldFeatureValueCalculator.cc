/*
 * OkapiBM25CrossFieldFeatureValueCalculator.cc
 *
 *  Created on: Jan 6, 2011
 *      Author: Chengnian SUN
 */

#include "OkapiBM25CrossFieldFeatureValueCalculator.h"
#include "../okapi/OkapiWrapper.h"

double OkapiBM25CrossFieldFeatureValueCalculator::weigh_term(const double idf,
    const double doc_tf, const double query_tf, const double doc_length,
    const double average_doc_length) const {
  // TODO implement OKAPI here...
  const double k1 = 1.2;
  const double b = 0.5;
  const double k3 = 0;

  return OkapiWrapper::compute_okapi_bm25(k1, k3, b, doc_tf, query_tf, idf,
      doc_length, average_doc_length);
}

OkapiBM25CrossFieldFeatureValueCalculator::OkapiBM25CrossFieldFeatureValueCalculator(
    const ReportBuckets& report_buckets,
    const SectionType::EnumSectionType query_type,
    const SectionType::EnumSectionType doc_type,
    const IDFCollectionType::EnumIDFCollectionType idf_type) :
    CrossFieldsFeatureValueCalculator(report_buckets, query_type, doc_type,
        idf_type) {

}

OkapiBM25CrossFieldFeatureValueCalculator::~OkapiBM25CrossFieldFeatureValueCalculator() {
}
