/*
 * AbstractFeatureValueCalculator.h
 *
 *  Created on: Jan 6, 2011
 *      Author: Chengnian SUN
 */

#ifndef ABSTRACTFEATUREVALUECALCULATOR_H_
#define ABSTRACTFEATUREVALUECALCULATOR_H_

class AbstractBugReport;
class ReportBuckets;
class DuplicateBugReport;

class AbstractFeatureValueCalculator {
private:

  const ReportBuckets& m_report_buckets;

protected:

  inline const ReportBuckets& get_report_buckets() const {
    return this->m_report_buckets;
  }

public:

  virtual double compute_feature_value(const AbstractBugReport& query,
      const AbstractBugReport& report) = 0;

  AbstractFeatureValueCalculator(const ReportBuckets& report_buckets);

  virtual ~AbstractFeatureValueCalculator();
};

#endif /* ABSTRACTFEATUREVALUECALCULATOR_H_ */
