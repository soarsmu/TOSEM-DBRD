/*
 * BM25FParameter.h
 *
 *  Created on: Jan 12, 2011
 *      Author: Chengnian SUN
 */

#ifndef BM25FPARAMETER_H_
#define BM25FPARAMETER_H_

#include <cassert>
#include <cstdio>

#include "../util/ParameterPair.h"

class BM25FParameter {

private:

  ParameterPair m_total_weight;

  ParameterPair m_summary_weight;

  ParameterPair m_summary_b;

  ParameterPair m_description_weight;

  ParameterPair m_description_b;

  ParameterPair m_k1;

  ParameterPair m_k3;

  float confine_b(const float b) const;

public:

  /**
   * total weight
   */
  float get_total_weight() const;

  bool is_total_weight_fixed() const;

  void increase_total_weight(const float delta);

  /**
   * summary weight
   */
  float get_summary_weight() const;

  bool is_summary_weight_fixed() const;

  void increase_summary_weight(const float delta);

  /**
   * summary b
   */
  float get_summary_b() const;

  bool is_summary_b_fixed() const;

  void increase_summary_b(const float delta);

  /**
   * description weight
   */
  float get_description_weight() const;

  bool is_description_weight_fixed() const;

  void increase_description_weight(const float delta);

  /**
   * description b
   */
  float get_description_b() const;

  bool is_description_b_fixed() const;

  void increase_description_b(const float delta);

  /**
   * k1
   */
  float get_k1() const;

  bool is_k1_fixed() const;

  void increase_k1(const float delta);

  /**
   * k3
   */
  float get_k3() const;

  bool is_k3_fixed() const;

  void increase_k3(const float delta);

  void set_k3(const float value, const bool fixed);

  void initialize(const float total_weight, const bool total_weight_fixed,
      const float k1, const bool k1_fixed, const float summary_weight,
      const bool summary_weight_fixed, const
      float summary_b, const bool summary_b_fixed,
      const float description_weight, const bool description_weight_fixed,
      const float description_b, const bool description_b_fixed,
      const float k3, const bool k3_fixed);

  void print(FILE* file, const char* title) const;
};

///////////////////////////////////////////////////////////////////////////////

inline float BM25FParameter::confine_b(const float b) const {
  if (b > 1) {
    return 1;
  } else if (b < 0) {
    return 0;
  } else {
    return b;
  }
}

/**
 * total weight
 */
inline float BM25FParameter::get_total_weight() const {
  return this->m_total_weight.value;
}

inline bool BM25FParameter::is_total_weight_fixed() const {
  return this->m_total_weight.fixed;
}

inline void BM25FParameter::increase_total_weight(const float delta) {
  assert(!this->m_total_weight.fixed);
  assert(delta == delta);
  this->m_total_weight.value += delta;
  assert(this->m_total_weight.value == this->m_total_weight.value);
  if (this->m_total_weight.value < 0) {
    this->m_total_weight.value = 0;
  }
}

/**
 * summary weight
 */
inline float BM25FParameter::get_summary_weight() const {
  return this->m_summary_weight.value;
}

inline bool BM25FParameter::is_summary_weight_fixed() const {
  return this->m_summary_weight.fixed;
}

inline void BM25FParameter::increase_summary_weight(const float delta) {
  assert(!this->m_summary_weight.fixed);
  assert(delta == delta);
  this->m_summary_weight.value += delta;
  assert(this->m_summary_weight.value == this->m_summary_weight.value);
  if (this->m_summary_weight.value < 0) {
    this->m_summary_weight.value = 1;
  }
}

/**
 * summary b
 */
inline float BM25FParameter::get_summary_b() const {
  return this->m_summary_b.value;
}

inline bool BM25FParameter::is_summary_b_fixed() const {
  return this->m_summary_b.fixed;
}

inline void BM25FParameter::increase_summary_b(const float delta) {
  assert(!this->m_summary_b.fixed);
  assert(delta == delta);
  this->m_summary_b.value = this->confine_b(this->m_summary_b.value + delta);
  assert(this->m_summary_b.value == this->m_summary_b.value);
}

/**
 * description weight
 */
inline float BM25FParameter::get_description_weight() const {
  return this->m_description_weight.value;
}

inline bool BM25FParameter::is_description_weight_fixed() const {
  return this->m_description_weight.fixed;
}

inline void BM25FParameter::increase_description_weight(const float delta) {
  assert(!this->m_description_weight.fixed);
  assert(delta == delta);
  this->m_description_weight.value += delta;
  assert( this->m_description_weight.value == this->m_description_weight.value);

  if (this->m_description_weight.value < 0) {
    this->m_description_weight.value = 0.005;
  }
}

/**
 * description b
 */
inline float BM25FParameter::get_description_b() const {
  return this->m_description_b.value;
}

inline bool BM25FParameter::is_description_b_fixed() const {
  return this->m_description_b.fixed;
}

inline void BM25FParameter::increase_description_b(const float delta) {
  assert(!this->m_description_b.fixed);
  assert(delta == delta);
  this->m_description_b.value = this->confine_b(
      this->m_description_b.value + delta);
  assert(this->m_description_b.value == this->m_description_b.value);
}

/**
 * k1
 */
inline float BM25FParameter::get_k1() const {
  return this->m_k1.value;
}

inline bool BM25FParameter::is_k1_fixed() const {
  return this->m_k1.fixed;
}

inline void BM25FParameter::increase_k1(const float delta) {
  assert(!this->m_k1.fixed);
  this->m_k1.value += delta;
  assert(delta == delta);
  assert(this->m_k1.value == this->m_k1.value);
  assert(this->m_k1.value > 0);
}

/**
 * k3
 */
inline float BM25FParameter::get_k3() const {
  return this->m_k3.value;
}

inline bool BM25FParameter::is_k3_fixed() const {
  return this->m_k3.fixed;
}

inline void BM25FParameter::increase_k3(const float delta) {
  assert(!this->m_k3.fixed);
  this->m_k3.value += delta;
  if (this->m_k3.value < 0) {
    this->m_k3.value = 0;
  }
  assert(delta == delta);
  assert(this->m_k3.value == this->m_k3.value);
  assert(this->m_k3.value >= 0);
}

inline void BM25FParameter::set_k3(const float value, const bool fixed) {
  this->m_k3.set(value, fixed);

  this->m_summary_b.fixed = !fixed;
  this->m_summary_weight.fixed = !fixed;

  this->m_description_b.fixed = !fixed;
  this->m_description_weight.fixed = !fixed;

}

#endif /* BM25FPARAMETER_H_ */
