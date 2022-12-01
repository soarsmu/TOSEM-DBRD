/*
 * AbstractFeatureVectorCalculator.cc
 *
 *  Created on: Jan 7, 2011
 *      Author: Chengnian SUN
 */

#include "AbstractFeatureVectorCalculator.h"
#include "../util/MacroUtility.h"
#include "../feature-value-calculator/AbstractFeatureValueCalculator.h"

#include <cassert>

void AbstractFeatureVectorCalculator::start_to_use() {
  // do nothing.
}

void AbstractFeatureVectorCalculator::fill_feature_values(
    svm_node* feature_vector, const AbstractBugReport& query,
    const AbstractBugReport& report) {

  const unsigned feature_count = this->m_value_calculators.size();

  assert(feature_count == this->get_feature_count());

  for (unsigned i = 0; i < feature_count; i++) {
    AbstractFeatureValueCalculator* calculator = this->m_value_calculators[i];
    feature_vector[i].value = calculator->compute_feature_value(query, report);
  }

}

AbstractFeatureVectorCalculator::AbstractFeatureVectorCalculator(
    const ReportBuckets& report_buckets, const unsigned textual_feature_count,
    const unsigned surface_feature_count) :
    m_buckets(report_buckets), m_Textual_feature_count(textual_feature_count), m_Surface_feature_count(
        surface_feature_count) {
}

AbstractFeatureVectorCalculator::~AbstractFeatureVectorCalculator(void) {
  for (vector<AbstractFeatureValueCalculator*>::iterator iter =
      this->m_value_calculators.begin();
      iter != this->m_value_calculators.end(); iter++) {
    delete (*iter);
  }
}

void AbstractFeatureVectorCalculator::init_feature_vector_calculator() {
  if (!this->m_value_calculators.empty()) {
    ERROR_HERE("this->m_value_calculators should be empty.");
  }

  this->m_value_calculators = this->create_Textual_feature_vector_calculators();

  if (this->m_value_calculators.size() != this->get_Textual_feature_count()) {
    char message[100];
    sprintf(message,
        "textual calculator size(%u) != specified textual feature count(%u)",
        static_cast<unsigned>(this->m_value_calculators.size()),
        static_cast<unsigned>(this->get_Textual_feature_count()));
    ERROR_HERE(message);
  }

  const vector<AbstractFeatureValueCalculator*>& surface_calculators =
      this->create_Surface_feature_vector_calculators();
  if (surface_calculators.size() != this->get_Surface_feature_count()) {
    char message[100];
    sprintf(message,
        "surface calculator size(%u) != specified surface feature count(%u)",
        static_cast<unsigned>(surface_calculators.size()),
        static_cast<unsigned>(this->get_Surface_feature_count()));
    ERROR_HERE(message);
  }
  this->m_value_calculators.insert(this->m_value_calculators.end(),
      surface_calculators.begin(), surface_calculators.end());

  if (this->m_value_calculators.size() != this->get_feature_count()) {
    char message[100];
    sprintf(message, "value calculator size(%u) != specified feature count(%u)",
        static_cast<unsigned>(this->m_value_calculators.size()),
        static_cast<unsigned>(this->get_feature_count()));
    ERROR_HERE(message);
  }
}

