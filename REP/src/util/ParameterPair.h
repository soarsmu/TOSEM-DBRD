/*
 * ParameterPair.h
 *
 *  Created on: Jan 11, 2011
 *      Author: Chengnian SUN
 */

#ifndef PARAMETERPAIR_H_
#define PARAMETERPAIR_H_

struct ParameterPair {
  float value;
  bool fixed;

  void set(const float value, const bool fixed) {
    this->value = value;
    this->fixed = fixed;
  }
};

#endif /* PARAMETERPAIR_H_ */
