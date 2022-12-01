/*
 * Term.h
 *
 *  Created on: Feb 1, 2011
 *      Author: neo
 */

#ifndef TERM_H_
#define TERM_H_

template<typename T>
class TermTemplate {
private:

  int m_term_id;

  T m_term_frequency;

public:

  int get_tid() const {
    return m_term_id;
  }

  T get_term_frequency() const {
    return m_term_frequency;
  }

  TermTemplate(int term_id, T term_frequency) :
      m_term_id(term_id), m_term_frequency(term_frequency) {
  }

  ~TermTemplate() {
  }
};

typedef TermTemplate<int> Term;

typedef TermTemplate<float> PreciseTerm;

//class Term: public TermTemplate<int> {
//
//public:
//  Term(int term_id, int term_frequency) :
//      TermTemplate(term_id, term_frequency) {
//
//  }
//
//};
//
//class PreciseTerm: public TermTemplate<float> {
//public:
//  PreciseTerm(int term_id, float term_frequency) :
//      TermTemplate(term_id, term_frequency) {
//  }
//};

class StructuredTerm {
private:
  int m_term_id;

  int m_summary_tf;

  int m_description_tf;

public:
  inline int get_tid() const {
    return m_term_id;
  }

  inline int get_description_tf() const {
    return this->m_description_tf;
  }

  inline int get_summary_tf() const {
    return m_summary_tf;
  }

  inline StructuredTerm(int term_id, int summary_tf, int description_tf) :
      m_term_id(term_id), m_summary_tf(summary_tf), m_description_tf(
          description_tf) {
  }

  inline ~StructuredTerm() {
  }
};

#endif /* TERM_H_ */
