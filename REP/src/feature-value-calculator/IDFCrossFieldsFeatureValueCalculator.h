/*
 * IDFCrossFieldsFeatureValueCalculator.h
 *
 *  Created on: Jan 6, 2011
 *      Author: Chengnian SUN
 */

#ifndef IDFCROSSFIELDSFEATUREVALUECALCULATOR_H_
#define IDFCROSSFIELDSFEATUREVALUECALCULATOR_H_

#include "AbstractFeatureValueCalculator.h"
#include "CrossFieldsFeatureValueCalculator.h"

class IDFCrossFieldsFeatureValueCalculator: public CrossFieldsFeatureValueCalculator {
protected:
  virtual double weigh_term(const double idf, const double doc_tf,
      const double query_tf, const double doc_length,
      const double average_doc_length) const;
public:
  IDFCrossFieldsFeatureValueCalculator(const ReportBuckets& report_buckets,
      const SectionType::EnumSectionType query_type,
      const SectionType::EnumSectionType doc_type,
      const IDFCollectionType::EnumIDFCollectionType idf_type);
  virtual ~IDFCrossFieldsFeatureValueCalculator();
};

#endif /* IDFCROSSFIELDSFEATUREVALUECALCULATOR_H_ */
