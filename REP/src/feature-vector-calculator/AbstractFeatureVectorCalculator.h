#ifndef __ABSTRACT_FEATURE_VECTOR_CALCULATOR_H__
#define __ABSTRACT_FEATURE_VECTOR_CALCULATOR_H__

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>
using namespace std;

#include "../libsvm/svm.h"

class DuplicateBugReport;
class AbstractBugReport;
class AbstractFeatureValueCalculator;
class ReportBuckets;

class AbstractFeatureVectorCalculator {
private:

  vector<AbstractFeatureValueCalculator*> m_value_calculators;

  const ReportBuckets& m_buckets;

  const unsigned m_Textual_feature_count;

  const unsigned m_Surface_feature_count;

protected:

  const ReportBuckets& get_report_buckets() const;

  virtual vector<AbstractFeatureValueCalculator*> create_Textual_feature_vector_calculators() const = 0;

  virtual vector<AbstractFeatureValueCalculator*> create_Surface_feature_vector_calculators() const = 0;

public:

  unsigned get_Surface_feature_count() const;

  unsigned get_Textual_feature_count() const;

  unsigned int get_feature_count() const;

  // this method is called before detecting a duplicate report.
  virtual void start_to_use();

  void fill_feature_values(svm_node* feature_vector,
      const AbstractBugReport& query, const AbstractBugReport& report);

  void init_feature_vector_calculator();

  AbstractFeatureVectorCalculator(const ReportBuckets& report_buckets,
      const unsigned textual_feature_count,
      const unsigned surface_feature_count);

  virtual ~AbstractFeatureVectorCalculator(void);

};

inline const ReportBuckets& AbstractFeatureVectorCalculator::get_report_buckets() const {
  return this->m_buckets;
}

inline unsigned AbstractFeatureVectorCalculator::get_Surface_feature_count() const {
  return this->m_Surface_feature_count;
}

inline unsigned AbstractFeatureVectorCalculator::get_Textual_feature_count() const {
  return this->m_Textual_feature_count;
}

inline unsigned AbstractFeatureVectorCalculator::get_feature_count() const {
  return this->m_Textual_feature_count + this->m_Surface_feature_count;
}
#endif /*__ABSTRACT_FEATURE_VECTOR_CALCULATOR_H__*/
