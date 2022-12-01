/*
 * SurfaceWeight.h
 *
 *  Created on: Jan 11, 2011
 *      Author: Chengnian SUN
 */

#ifndef SURFACEWEIGHT_H_
#define SURFACEWEIGHT_H_

#include "../util/ParameterPair.h"

class SurfaceWeight {

private:

  //	ParameterPair m_unigram_weight;
  //
  //	ParameterPair m_bigram_weight;

  ParameterPair m_component_weight;

  ParameterPair m_sub_component_weight;

  ParameterPair m_report_type_weight;

  ParameterPair m_priority_weight;

  ParameterPair m_version_weight;

public:

  // ---- component weight
  inline double get_component_weight() const {
    return this->m_component_weight.value;
  }

  inline bool is_component_weight_fixed() const {
    return this->m_component_weight.fixed;
  }

  inline void increase_component_weight(const double delta) {
    assert(!this->m_component_weight.fixed);
    this->m_component_weight.value += delta;
    assert(delta == delta);
    assert(this->m_component_weight.value == this->m_component_weight.value);
  }

  // ---- sub-component weight
  inline double get_sub_component_weight() const {
    return this->m_sub_component_weight.value;
  }

  inline bool is_sub_component_weight_fixed() const {
    return this->m_sub_component_weight.fixed;
  }

  inline void increase_sub_component_weight(const double delta) {
    assert(!this->m_sub_component_weight.fixed);
    this->m_sub_component_weight.value += delta;
    assert(delta == delta);
    assert(
        this->m_sub_component_weight.value
            == this->m_sub_component_weight.value);
    if (this->m_sub_component_weight.value < 0) {
      this->m_sub_component_weight.value = 0;
    }
  }

  // ---- report type weight
  inline double get_report_type_weight() const {
    return this->m_report_type_weight.value;
  }

  inline bool is_report_type_weight_fixed() const {
    return this->m_report_type_weight.fixed;
  }

  inline void increase_report_type_weight(const double delta) {
    assert(!this->m_report_type_weight.fixed);
    this->m_report_type_weight.value += delta;
    assert(delta == delta);
    assert(
        this->m_report_type_weight.value == this->m_report_type_weight.value);
    if (this->m_report_type_weight.value < 0) {
      this->m_report_type_weight.value = 0;
    }
  }

  // ---- priority weight
  inline double get_priority_weight() const {
    return this->m_priority_weight.value;
  }

  inline bool is_priority_weight_fixed() const {
    return this->m_priority_weight.fixed;
  }

  inline void increase_priority_weight(const double delta) {
    assert(!this->m_priority_weight.fixed);
    this->m_priority_weight.value += delta;
    assert(delta == delta);
    assert(this->m_priority_weight.value == this->m_priority_weight.value);
    if (this->m_priority_weight.value < 0) {
      this->m_priority_weight.value = 0;
    }
  }

  // ---- version weight
  inline double get_version_weight() const {
    assert(this->m_version_weight.value == this->m_version_weight.value);
    return this->m_version_weight.value;
  }

  inline bool is_version_weight_fixed() const {
    return this->m_version_weight.fixed;
  }

  inline void increase_version_weight(const double delta) {
    assert(!this->m_version_weight.fixed);
    this->m_version_weight.value += delta;
    assert(delta == delta);
    assert(this->m_version_weight.value == this->m_version_weight.value);
//		if (this->m_version_weight.value < 0) {
//			this->m_version_weight.value = 0;
//		}
  }

  inline void initialize(const double component_weight,
      const bool component_weight_fixed, const double sub_component_weight,
      const bool sub_component_weight_fixed, const double report_type_weight,
      const bool report_type_weight_fixed, const double priority_weight,
      const bool priority_weight_fixed, const double version_weight,
      const bool version_weight_fixed) {
    this->m_component_weight.set(component_weight, component_weight_fixed);
    this->m_sub_component_weight.set(sub_component_weight,
        sub_component_weight_fixed);
    this->m_report_type_weight.set(report_type_weight,
        report_type_weight_fixed);
    this->m_priority_weight.set(priority_weight, priority_weight_fixed);
    this->m_version_weight.set(version_weight, version_weight_fixed);
  }

  inline void print(FILE* file) const {
    fprintf(file, "\nSurface weights...\n");

    //		fprintf(stdout, "[unigram weight, fixed = %d] = %f\n", this->m_unigram_weight.fixed, this->m_unigram_weight.value);
    //		fprintf(stdout, "[bigram weight, fixed = %d] = %f\n", this->m_bigram_weight.fixed, this->m_bigram_weight.value);
    fprintf(file, "[component weight, fixed = %d] = %f\n",
        this->m_component_weight.fixed, this->m_component_weight.value);
    fprintf(file, "[sub component weight, fixed = %d] = %f\n",
        this->m_sub_component_weight.fixed, this->m_sub_component_weight.value);
    fprintf(file, "[report type weight, fixed = %d] = %f\n",
        this->m_report_type_weight.fixed, this->m_report_type_weight.value);
    fprintf(file, "[priority weight, fixed = %d] = %f\n",
        this->m_priority_weight.fixed, this->m_priority_weight.value);
    fprintf(file, "[version weight, fixed = %d] = %f\n",
        this->m_version_weight.fixed, this->m_version_weight.value);
  }

};

#endif /* SURFACEWEIGHT_H_ */
