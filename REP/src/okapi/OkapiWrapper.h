/*
 * OkapiWrapper.h
 *
 *  Created on: Jan 3, 2011
 *      Author: Chengnian SUN
 */

#ifndef __OKAPI__WRAPPER_H__
#define __OKAPI__WRAPPER_H__

#include <cassert>
#include <cstdio>
#include <iostream>

#include "../report-model/Section.h"
#include "../tfidf/IDFCollection.h"

using namespace std;

class BM25FParameter;

class OkapiWrapper {

public:

  struct LengthInfo {

    unsigned summary_length;

    unsigned description_length;

    LengthInfo(unsigned summ_len, unsigned desc_len) :
        summary_length(summ_len), description_length(desc_len) {
    }

  };

  struct AverageLengthInfo {

    float summary_average_length;

    float description_average_length;

    AverageLengthInfo(float sum_leng, float desc_leng) :
        summary_average_length(sum_leng), description_average_length(desc_leng) {
    }

  };

public:

  static float compute_tf_per_field_for_bm25f(const int tf, const float weight,
      const float b, const int length, const float average_length);

  static float compute_tf_component_for_bm25f(const int summary_tf,
      const int desc_tf, const BM25FParameter& parameter,
      const LengthInfo length_info,
      const AverageLengthInfo average_length_info);

  static float compute_term_weight_for_bm25f(const int summary_tf,
      const int desc_tf, const BM25FParameter& parameter,
      const LengthInfo length_info,
      const AverageLengthInfo average_length_info);

  static float compute_query_weight_for_bm25f(const BM25FParameter& parameter,
      const int summary_tf_query, const int desc_tf_query);

protected:

  template<typename TermWeightingFunc>
  static double skeleton(IDFCollection* idf_collection,
      const AverageLengthInfo& average_length_info,
      const StructuredSection& query_section,
      const StructuredSection& base_section,
      const struct BM25FParameter& parameter) {
    TermWeightingFunc term_weighting_func;
    //		const StructuredSection& query_section = query_report.get_Unigram_structured_section();
    //		const StructuredSection& base_section = base_report.get_Unigram_structured_section();

    const LengthInfo length_info(base_section.get_summary_length(),
        base_section.get_description_length());

    double result = 0;

    const vector<StructuredTerm>& query_terms = query_section.get_terms();
    const vector<StructuredTerm>& base_terms = base_section.get_terms();

    const unsigned query_term_count = query_terms.size();
    const unsigned base_term_count = base_terms.size();
    unsigned query_index = 0;
    unsigned base_index = 0;

    while (query_index < query_term_count && base_index < base_term_count) {
      const StructuredTerm& query_term = query_terms[query_index];
      const StructuredTerm& base_term = base_terms[base_index];

      const int query_tid = query_term.get_tid();
      const int base_tid = base_term.get_tid();
      if (query_tid > base_tid) {
        base_index++;
      } else if (query_tid < base_tid) {
        query_index++;
      } else {
        double idf_component = idf_collection->get_idf(query_tid);
        const int base_summary_tf = base_term.get_summary_tf();
        const int base_desc_tf = base_term.get_description_tf();
        const int query_summary_tf = query_term.get_summary_tf();
        const int query_desc_tf = query_term.get_description_tf();

        result += term_weighting_func(idf_component, base_summary_tf,
            base_desc_tf, parameter, length_info, average_length_info,
            query_summary_tf, query_desc_tf);

        base_index++;
        query_index++;
      }
    }
    return result;
  }

public:

  static double
  compute_bm25f_derivative_wrt_k(IDFCollection* idf_collection,
      const AverageLengthInfo& average_length_info,
      const StructuredSection& query_section,
      const StructuredSection& base_section,
      const struct BM25FParameter& parameter);

  static double compute_bf25f_derivative_wrt_summary_weight(
      IDFCollection* idf_collection,
      const AverageLengthInfo& average_length_info,
      const StructuredSection& query_section,
      const StructuredSection& base_section,
      const struct BM25FParameter& parameter);

  static double compute_bf25f_derivative_wrt_description_weight(
      IDFCollection* idf_collection,
      const AverageLengthInfo& average_length_info,
      const StructuredSection& query_section,
      const StructuredSection& base_section,
      const struct BM25FParameter& parameter);

  static double compute_bf25f_derivative_wrt_summary_b(
      IDFCollection* idf_collection,
      const AverageLengthInfo& average_length_info,
      const StructuredSection& query_section,
      const StructuredSection& base_section,
      const struct BM25FParameter& parameter);

  static double compute_bf25f_derivative_wrt_description_b(
      IDFCollection* idf_collection,
      const AverageLengthInfo& average_length_info,
      const StructuredSection& query_section,
      const StructuredSection& base_section,
      const struct BM25FParameter& parameter);

  static double compute_bf25f_derivative_wrt_k3(IDFCollection* idf_collection,
      const AverageLengthInfo& average_length_info,
      const StructuredSection& query_section,
      const StructuredSection& base_section,
      const struct BM25FParameter& parameter);

  /**
   * BM25F, an extension of BM25 to support multi-field documents
   */
  static double compute_bm25f(IDFCollection* idf_collection,
      const AverageLengthInfo& average_length_info,
      const StructuredSection& query_section,
      const StructuredSection& base_section,
      const struct BM25FParameter& parameter);

  /**
   * Okapi BM25.
   *
   * this method seems not to be used widely. so please ignore it.
   */
  static double compute_okapi_bm25(double k1, double k3, double b,
      int base_term_tf, int query_term_tf, double idf, unsigned doc_length,
      double average_doc_length);

private:

  OkapiWrapper() {
  }

  virtual ~OkapiWrapper() {
  }
};

#endif /* __OKAPI__WRAPPER_H__ */
