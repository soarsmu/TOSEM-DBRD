/*
 * Section.cc
 *
 *  Created on: Mar 15, 2013
 *      Author: neo
 */
#include <cmath>

#include "Section.h"

unsigned Section::compute_length(const vector<Term>& terms) {
  unsigned length = 0;
  for (vector<Term>::const_iterator iter = terms.begin(), end = terms.end();
      iter != end; ++iter) {
    length += iter->get_term_frequency();
  }
  return length;
}

Section::Section(const vector<Term>& terms) :
    m_terms(terms), m_length(Section::compute_length(terms)) {
}

//double Section::compute_vector_length(const vector<Term>& terms) {
//  unsigned length = 0;
//  for (vector<Term>::const_iterator iter = terms.begin(), end = terms.end();
//      iter != end; ++iter) {
//    length += tf * tf;
//  }
//  return sqrt(length);
//}
