/*
 * OkapiWrapper.cc
 *
 *  Created on: Jan 11, 2011
 *      Author: Chengnian SUN
 */

#include "OkapiWrapper.h"
#include "BM25FParameter.h"

struct BM25F_Derivative_wrt_K {
  double operator ()(const double idf, const int summary_tf, const int desc_tf,
      const BM25FParameter& parameter,
      const OkapiWrapper::LengthInfo length_info,
      const OkapiWrapper::AverageLengthInfo average_length_info, const int,
      const int) {

    assert(parameter.get_k3() == 0 && parameter.is_k3_fixed());

    double tf_component = OkapiWrapper::compute_tf_component_for_bm25f(
        summary_tf, desc_tf, parameter, length_info, average_length_info);

    return -(idf * tf_component / pow((parameter.get_k1() + tf_component), 2));
  }
};

double OkapiWrapper::compute_bm25f_derivative_wrt_k(
    IDFCollection* idf_collection, const AverageLengthInfo& average_length_info,
    const StructuredSection& query_section,
    const StructuredSection& base_section,
    const struct BM25FParameter& parameter) {

  return skeleton<BM25F_Derivative_wrt_K>(idf_collection, average_length_info,
      query_section, base_section, parameter);
}

struct BM25F_Derivative_wrt_summary_weight {
  double operator ()(const double idf, const int summary_tf, const int desc_tf,
      const BM25FParameter& parameter,
      const OkapiWrapper::LengthInfo length_info,
      const OkapiWrapper::AverageLengthInfo average_length_info, const int,
      const int) {

    assert(parameter.get_k3() == 0 && parameter.is_k3_fixed());

    if (summary_tf == 0) {
      return 0;
    }

    const double tf_component = OkapiWrapper::compute_tf_component_for_bm25f(
        summary_tf, desc_tf, parameter, length_info, average_length_info);

    //			return (-idf * tf_component / pow((parameter.k1 + tf_component), 2));
    const double beta_summary = 1 - parameter.get_summary_b()
        + parameter.get_summary_b() * length_info.summary_length
            / average_length_info.summary_average_length;

    const double denominator = pow((parameter.get_k1() + tf_component), 2)
        * beta_summary;

    const double numerator = idf * parameter.get_k1() * summary_tf;

    return numerator / denominator;
  }
};

double OkapiWrapper::compute_bf25f_derivative_wrt_summary_weight(
    IDFCollection* idf_collection, const AverageLengthInfo& average_length_info,
    const StructuredSection& query_section,
    const StructuredSection& base_section,
    const struct BM25FParameter& parameter) {
  return skeleton<BM25F_Derivative_wrt_summary_weight>(idf_collection,
      average_length_info, query_section, base_section, parameter);
}

struct BM25F_Derivative_wrt_description_weight {
  double operator ()(const double idf, const int summary_tf, const int desc_tf,
      const BM25FParameter& parameter,
      const OkapiWrapper::LengthInfo length_info,
      const OkapiWrapper::AverageLengthInfo average_length_info, const int,
      const int) {

    assert(parameter.get_k3() == 0 && parameter.is_k3_fixed());

    if (desc_tf == 0) {
      return 0;
    }

    const double tf_component = OkapiWrapper::compute_tf_component_for_bm25f(
        summary_tf, desc_tf, parameter, length_info, average_length_info);

    //			return (-idf * tf_component / pow((parameter.k1 + tf_component), 2));
    const double beta_description = 1 - parameter.get_description_b()
        + parameter.get_description_b() * length_info.description_length
            / average_length_info.description_average_length;

    const double denominator = pow((parameter.get_k1() + tf_component), 2)
        * beta_description;

    const double numerator = idf * parameter.get_k1() * desc_tf;

    return numerator / denominator;
  }
};

double OkapiWrapper::compute_bf25f_derivative_wrt_description_weight(
    IDFCollection* idf_collection, const AverageLengthInfo& average_length_info,
    const StructuredSection& query_section,
    const StructuredSection& base_section,
    const struct BM25FParameter& parameter) {
  return skeleton<BM25F_Derivative_wrt_description_weight>(idf_collection,
      average_length_info, query_section, base_section, parameter);
}

struct BM25F_Derivative_wrt_summary_b {
  double operator ()(const double idf, const int summary_tf, const int desc_tf,
      const BM25FParameter& parameter,
      const OkapiWrapper::LengthInfo length_info,
      const OkapiWrapper::AverageLengthInfo average_length_info, const int,
      const int) {

    assert(parameter.get_k3() == 0 && parameter.is_k3_fixed());

    if (summary_tf == 0) {
      return 0;
    }

    const double tf_component = OkapiWrapper::compute_tf_component_for_bm25f(
        summary_tf, desc_tf, parameter, length_info, average_length_info);

    const double beta_summary = 1 - parameter.get_summary_b()
        + parameter.get_summary_b() * length_info.summary_length
            / average_length_info.summary_average_length;

    const double denominator = pow((parameter.get_k1() + tf_component), 2)
        * beta_summary * beta_summary;

    const double numerator = idf * parameter.get_k1()
        * parameter.get_summary_weight() * summary_tf
        * (1
            - length_info.summary_length
                / average_length_info.summary_average_length);

    return numerator / denominator;
  }
};

double OkapiWrapper::compute_bf25f_derivative_wrt_summary_b(
    IDFCollection* idf_collection, const AverageLengthInfo& average_length_info,
    const StructuredSection& query_section,
    const StructuredSection& base_section,
    const struct BM25FParameter& parameter) {
  return skeleton<BM25F_Derivative_wrt_summary_b>(idf_collection,
      average_length_info, query_section, base_section, parameter);
}

struct BM25F_Derivative_wrt_description_b {
  double operator ()(const double idf, const int summary_tf, const int desc_tf,
      const BM25FParameter& parameter,
      const OkapiWrapper::LengthInfo length_info,
      const OkapiWrapper::AverageLengthInfo average_length_info, const int,
      const int) {

    assert(parameter.get_k3() == 0 && parameter.is_k3_fixed());

    if (desc_tf == 0) {
      return 0;
    }
    const double tf_component = OkapiWrapper::compute_tf_component_for_bm25f(
        summary_tf, desc_tf, parameter, length_info, average_length_info);

    const double beta_description = 1 - parameter.get_description_b()
        + parameter.get_description_b() * length_info.description_length
            / average_length_info.description_average_length;
    assert(average_length_info.description_average_length > 0);
    const double denominator = pow((parameter.get_k1() + tf_component), 2)
        * beta_description * beta_description;

    const double numerator = idf * parameter.get_k1()
        * parameter.get_description_b() * desc_tf
        * (1
            - length_info.description_length
                / average_length_info.description_average_length);

    assert(denominator > 0);
    return numerator / denominator;
  }
};

double OkapiWrapper::compute_bf25f_derivative_wrt_description_b(
    IDFCollection* idf_collection, const AverageLengthInfo& average_length_info,
    const StructuredSection& query_section,
    const StructuredSection& base_section,
    const struct BM25FParameter& parameter) {
  return skeleton<BM25F_Derivative_wrt_description_b>(idf_collection,
      average_length_info, query_section, base_section, parameter);
}

struct BM25F_Derivative_wrt_k3 {
  double operator ()(const double idf, const int summary_tf, const int desc_tf,
      const BM25FParameter& parameter,
      const OkapiWrapper::LengthInfo length_info,
      const OkapiWrapper::AverageLengthInfo average_length_info,
      const int query_summary_tf, const int query_desc_tf) {

    assert(parameter.is_description_b_fixed());
    assert(parameter.is_description_weight_fixed());
    assert(parameter.is_summary_b_fixed());
    assert(parameter.is_summary_weight_fixed());
    assert(!parameter.is_k3_fixed());

    double tf_component = OkapiWrapper::compute_tf_component_for_bm25f(
        summary_tf, desc_tf, parameter, length_info, average_length_info);

    const double k3 = parameter.get_k3();
    /*
     * tf = summary_wegith * summary_tf + desc_weight * desc_tf;
     *
     * query weight = (k3 + 1) * tf / (k3 + tf)
     */
    const double query_tf_component = parameter.get_summary_weight()
        * query_summary_tf + parameter.get_description_weight() * query_desc_tf;

    assert(query_summary_tf + query_desc_tf > 0);
    assert(
        parameter.get_summary_weight( ) + parameter.get_description_weight() > 0);
    assert(query_tf_component > 0);
    assert(k3 == k3);
    //		double query_weight = (k3 + 1) * query_tf_component / (k3 + query_tf_component);
    const double query_weight = query_tf_component * (query_tf_component - 1)
        / (pow(k3 + query_tf_component, 2));

    double result = idf * tf_component / (parameter.get_k1() + tf_component)
        * query_weight;
    assert(query_weight == query_weight);
    assert(idf == idf);
    assert(tf_component == tf_component);
    assert(parameter.get_k1() >= 0);
    assert((parameter.get_k1() + tf_component) != 0);
    assert(result == result);

    return result;

  }
};

double OkapiWrapper::compute_bf25f_derivative_wrt_k3(
    IDFCollection* idf_collection, const AverageLengthInfo& average_length_info,
    const StructuredSection& query_section,
    const StructuredSection& base_section,
    const struct BM25FParameter& parameter) {
  return skeleton<BM25F_Derivative_wrt_k3>(idf_collection, average_length_info,
      query_section, base_section, parameter);
}

float OkapiWrapper::compute_query_weight_for_bm25f(
    const BM25FParameter& parameter, const int summary_tf_query,
    const int desc_tf_query) {
  const float k3 = parameter.get_k3();
  if (k3 == 0) {
    return 1;
  }
  /*
   * tf = summary_wegith * summary_tf + desc_weight * desc_tf;
   *
   * query weight = (k3 + 1) * tf / (k3 + tf)
   */
  const double query_tf_component = parameter.get_summary_weight()
      * summary_tf_query + parameter.get_description_weight() * desc_tf_query;
  assert(query_tf_component > 0);
  assert(k3 == k3);
  return (k3 + 1) * query_tf_component / (k3 + query_tf_component);
}

float OkapiWrapper::compute_term_weight_for_bm25f(const int summary_tf,
    const int desc_tf, const BM25FParameter& parameter,
    const LengthInfo length_info, const AverageLengthInfo average_length_info) {
  float tf_component = OkapiWrapper::compute_tf_component_for_bm25f(summary_tf,
      desc_tf, parameter, length_info, average_length_info);

  assert(tf_component == tf_component);
  assert(parameter.get_k1() >= 0);
  assert((parameter.get_k1() + tf_component) != 0);

  return tf_component / (parameter.get_k1() + tf_component);
}

struct BM25FTermWeightingMeasure {
  double operator ()(const double idf, const int summary_tf, const int desc_tf,
      const BM25FParameter& parameter,
      const OkapiWrapper::LengthInfo length_info,
      const OkapiWrapper::AverageLengthInfo average_length_info,
      const int query_summary_tf, const int query_desc_tf) {

//    float tf_component = OkapiWrapper::compute_tf_component_for_bm25f(
//        summary_tf, desc_tf, parameter, length_info, average_length_info);
    float term_weight = OkapiWrapper::compute_term_weight_for_bm25f(summary_tf,
        desc_tf, parameter, length_info, average_length_info);
    float query_weight = OkapiWrapper::compute_query_weight_for_bm25f(parameter,
        query_summary_tf, query_desc_tf);
//    const double k3 = parameter.get_k3();
//    if (k3 == 0) {
//      query_weight = 1;
//    } else {
//      /*
//       * tf = summary_wegith * summary_tf + desc_weight * desc_tf;
//       *
//       * query weight = (k3 + 1) * tf / (k3 + tf)
//       */
//      const double query_tf_component = parameter.get_summary_weight()
//          * query_summary_tf
//          + parameter.get_description_weight() * query_desc_tf;
//      assert(query_tf_component > 0);
//      assert(k3 == k3);
//      query_weight = (k3 + 1) * query_tf_component / (k3 + query_tf_component);
//    }
//    float result = idf * tf_component / (parameter.get_k1() + tf_component)
//        * query_weight;
    float result = idf * term_weight * query_weight;
    assert(query_weight == query_weight);
    assert(idf == idf);
    assert(result == result);

    return result;
  }
};

/**
 * BM25F, an extension of BM25 to support multi-field documents
 */
double OkapiWrapper::compute_bm25f(IDFCollection* idf_collection,
    const AverageLengthInfo& average_length_info,
    const StructuredSection& query_section,
    const StructuredSection& base_section,
    const struct BM25FParameter& parameter) {

  return skeleton<BM25FTermWeightingMeasure>(idf_collection,
      average_length_info, query_section, base_section, parameter);
}

/**
 * Okapi BM25.
 */
double OkapiWrapper::compute_okapi_bm25(double k1, double k3, double b,
    int base_term_tf, int query_term_tf, double idf, unsigned doc_length,
    double average_doc_length) {

  double doc_weight_numerator = (k1 + 1) * base_term_tf;
  double doc_weight_denominator = k1
      * ((1 - b) + b * (doc_length / average_doc_length)) + base_term_tf;

  double query_weight_numerator = (k3 + 1) * query_term_tf;
  double query_weight_denominator = k3 + query_term_tf;

  double result = idf * (doc_weight_numerator / doc_weight_denominator)
      * (query_weight_numerator / query_weight_denominator);
  return result;
}

float OkapiWrapper::compute_tf_component_for_bm25f(const int summary_tf,
    const int desc_tf, const BM25FParameter& parameter,
    const LengthInfo length_info, const AverageLengthInfo average_length_info) {
  float summary_tf_component;

  assert(summary_tf != 0 || desc_tf != 0);

  if (summary_tf == 0) {
    summary_tf_component = 0;
  } else {
    summary_tf_component = compute_tf_per_field_for_bm25f(summary_tf,
        parameter.get_summary_weight(), parameter.get_summary_b(),
        length_info.summary_length, average_length_info.summary_average_length);
  }

  float desc_tf_component;
  if (desc_tf == 0) {
    desc_tf_component = 0;
  } else {
    desc_tf_component = compute_tf_per_field_for_bm25f(desc_tf,
        parameter.get_description_weight(), parameter.get_description_b(),
        length_info.description_length,
        average_length_info.description_average_length);
  }

  return summary_tf_component + desc_tf_component;
}

float OkapiWrapper::compute_tf_per_field_for_bm25f(const int tf,
    const float weight, const float b, const int length,
    const float average_length) {
  const float numerator = weight * tf;
  const float denominator = 1 - b + b * length / average_length;
  assert(denominator == denominator);
  assert(denominator != 0);
  const float normalized_tf = numerator / denominator;
  assert(normalized_tf == normalized_tf);
  return normalized_tf;
}
