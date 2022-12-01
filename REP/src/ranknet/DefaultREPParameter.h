/*
 * DefaultModelParameter.h
 *
 *  Created on: Jan 12, 2011
 *      Author: Chengnian SUN
 */

#ifndef DEFAULTMODELPARAMETER_H_
#define DEFAULTMODELPARAMETER_H_

#include <string>
using namespace std;

class DefaultREPParameter {
private:

  double m_unigram_weight;
  bool m_unigram_weight_fixed;

  double m_bigram_weight;
  bool m_bigram_weight_fixed;

  double m_k1;
  bool m_k1_fixed;

  double m_summary_weight;
  bool m_summary_weight_fixed;

  double m_summary_b;
  bool m_summary_b_fixed;

  double m_description_weight;
  bool m_description_weight_fixed;

  double m_description_b;
  bool m_description_b_fixed;

  double m_k3;
  bool m_k3_fixed;

  double m_component_weight;
  bool m_component_weight_fixed;

  double m_sub_component_weight;
  bool m_sub_component_weight_fixed;

  double m_report_type_weight;
  bool m_report_type_weight_fixed;

  double m_priority_weight;
  bool m_priority_weight_fixed;

  double m_version_weight;
  bool m_version_weight_fixed;

  unsigned m_count_of_irrelevant_reports_per_query;
  unsigned m_max_query_count;

public:

  double get_unigram_weight() const;

  bool is_unigram_weight_fixed() const;

  double get_bigram_weight() const;

  bool is_bigram_weight_fixed() const;

  double get_k1() const;

  bool is_k1_fixed() const;

  double get_summary_weight() const;

  bool is_summary_weight_fixed() const;

  double get_summary_b() const;

  bool is_summary_b_fixed() const;

  double get_description_weight() const;

  bool is_description_weight_fixed() const;

  double get_description_b() const;

  bool is_description_b_fixed() const;

  double get_k3() const;

  bool is_k3_fixed() const;

  double get_component_weight() const;

  bool is_component_weight_fixed() const;

  double get_sub_component_weight() const;

  bool is_sub_component_weight_fixed() const;

  double get_report_type_weight() const;

  bool is_report_type_weight_fixed() const;

  double get_priority_weight() const;

  bool is_priority_weight_fixed() const;

  double get_version_weight() const;

  bool is_version_weight_fixed() const;

  unsigned get_count_of_irrelevant_reports_per_query() const;

  unsigned get_max_query_count() const;

  DefaultREPParameter(const string& config_file_path);

};

///////////////////////////////////////////////////////////////////////////////

inline double DefaultREPParameter::get_unigram_weight() const {
  return this->m_unigram_weight;
}

inline bool DefaultREPParameter::is_unigram_weight_fixed() const {
  return this->m_unigram_weight_fixed;
}

inline double DefaultREPParameter::get_bigram_weight() const {
  return this->m_bigram_weight;
}

inline bool DefaultREPParameter::is_bigram_weight_fixed() const {
  return this->m_bigram_weight_fixed;
}

inline double DefaultREPParameter::get_k1() const {
  return this->m_k1;
}

inline bool DefaultREPParameter::is_k1_fixed() const {
  return this->m_k1_fixed;
}

inline double DefaultREPParameter::get_summary_weight() const {
  return this->m_summary_weight;
}

inline bool DefaultREPParameter::is_summary_weight_fixed() const {
  return this->m_summary_weight_fixed;
}

inline double DefaultREPParameter::get_summary_b() const {
  return this->m_summary_b;
}

inline bool DefaultREPParameter::is_summary_b_fixed() const {
  return this->m_summary_b_fixed;
}

inline double DefaultREPParameter::get_description_weight() const {
  return this->m_description_weight;
}

inline bool DefaultREPParameter::is_description_weight_fixed() const {
  return this->m_description_weight_fixed;
}

inline double DefaultREPParameter::get_description_b() const {
  return this->m_description_b;
}

inline bool DefaultREPParameter::is_description_b_fixed() const {
  return this->m_description_b_fixed;
}

inline double DefaultREPParameter::get_k3() const {
  return this->m_k3;
}

inline bool DefaultREPParameter::is_k3_fixed() const {
  return this->m_k3_fixed;
}

inline double DefaultREPParameter::get_component_weight() const {
  return this->m_component_weight;
}

inline bool DefaultREPParameter::is_component_weight_fixed() const {
  return this->m_component_weight_fixed;
}

inline double DefaultREPParameter::get_sub_component_weight() const {
  return this->m_sub_component_weight;
}

inline bool DefaultREPParameter::is_sub_component_weight_fixed() const {
  return this->m_sub_component_weight_fixed;
}

inline double DefaultREPParameter::get_report_type_weight() const {
  return this->m_report_type_weight;
}

inline bool DefaultREPParameter::is_report_type_weight_fixed() const {
  return this->m_report_type_weight_fixed;
}

inline double DefaultREPParameter::get_priority_weight() const {
  return this->m_priority_weight;
}

inline bool DefaultREPParameter::is_priority_weight_fixed() const {
  return this->m_priority_weight_fixed;
}

inline double DefaultREPParameter::get_version_weight() const {
  return this->m_version_weight;
}

inline bool DefaultREPParameter::is_version_weight_fixed() const {
  return this->m_version_weight_fixed;
}

inline unsigned DefaultREPParameter::get_count_of_irrelevant_reports_per_query() const {
  return this->m_count_of_irrelevant_reports_per_query;
}

inline unsigned DefaultREPParameter::get_max_query_count() const {
  return this->m_max_query_count;
}

#endif /* DEFAULTMODELPARAMETER_H_ */
