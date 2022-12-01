/*
 * OkapiBM25CrossFieldFeatureValueCalculator.h
 *
 *  Created on: Jan 6, 2011
 *      Author: Chengnian SUN
 */

#ifndef OKAPIBM25CROSSFIELDFEATUREVALUECALCULATOR_H_
#define OKAPIBM25CROSSFIELDFEATUREVALUECALCULATOR_H_

#include "CrossFieldsFeatureValueCalculator.h"

class OkapiBM25CrossFieldFeatureValueCalculator: public CrossFieldsFeatureValueCalculator {
private:
  // TODO, I need the bm25 parameters here...

protected:
  virtual double weigh_term(const double idf, const double doc_tf,
      const double query_tf, const double doc_length,
      const double average_doc_length) const;
public:

  OkapiBM25CrossFieldFeatureValueCalculator(const ReportBuckets& report_buckets,
      const SectionType::EnumSectionType query_type,
      const SectionType::EnumSectionType doc_type,
      const IDFCollectionType::EnumIDFCollectionType idf_type);

  virtual ~OkapiBM25CrossFieldFeatureValueCalculator();

};

#endif /* OKAPIBM25CROSSFIELDFEATUREVALUECALCULATOR_H_ */
