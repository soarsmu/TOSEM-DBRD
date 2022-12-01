/*
 * CrossFieldsFeatureValueCalculator.h
 *
 *  Created on: Jan 6, 2011
 *      Author: Chengnian SUN
 */

#ifndef CROSSFIELDSFEATUREVALUECALCULATOR_H_
#define CROSSFIELDSFEATUREVALUECALCULATOR_H_

#include "../report-model/SectionType.h"
#include "../tfidf/IDFCollectionType.h"

#include "AbstractFeatureValueCalculator.h"

class CrossFieldsFeatureValueCalculator: public AbstractFeatureValueCalculator {

private:

  const enum SectionType::EnumSectionType m_query_type;

  const enum SectionType::EnumSectionType m_doc_type;

  const enum IDFCollectionType::EnumIDFCollectionType m_idf_type;

protected:

  virtual double weigh_term(const double idf, const double doc_tf,
      const double query_tf, const double doc_length,
      const double average_doc_length) const = 0;

public:

  virtual double compute_feature_value(const AbstractBugReport& query,
      const AbstractBugReport& report);

  CrossFieldsFeatureValueCalculator(const ReportBuckets& report_buckets,
      const SectionType::EnumSectionType query_type,
      const SectionType::EnumSectionType doc_type,
      const IDFCollectionType::EnumIDFCollectionType idf_type);

  virtual ~CrossFieldsFeatureValueCalculator();
};

#endif /* CROSSFIELDSFEATUREVALUECALCULATOR_H_ */
