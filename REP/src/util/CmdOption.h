/*
 * CmdOption.h
 *
 *  Created on: Nov 8, 2010
 *      Author: Chengnian Sun
 */

#ifndef CMDOPTION_H_
#define CMDOPTION_H_

#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "../detector/DetectorFactory.h"
#include "../detection-model/IndexingType.h"
#include "../feature-vector-calculator/FeatureVectorCalculatorFactory.h"
#include "../plain-similarity-measures/PlainSimilarityMeasureFactory.h"

namespace po = boost::program_options;
using namespace std;

using std::string;
using std::vector;
/**
 * this is only for releasing to public.
 */
#define ONLY_ENABLE_RANKNET

class CmdOption {
private:

  static CmdOption* INSTANCE;

  unsigned m_training_duplicates;

  int m_iteration;

  int m_top_k;

  unsigned m_indexing_top_cosine_number;

  unsigned m_indexing_summary_weight;

  FeatureVectorCalculatorFactory::EnumFeatureCalculatorType m_feature_calculator_type;

  PlainSimilarityMeasureFactory::SimilarityMeasureType m_plain_measure_type;

  string m_dataset;

  string m_timestamp_file;

  string m_train_file;

  string m_project_name;

  string m_parsing_error;

//  string m_recommendation_file;

  DetectorFactory::EnumDetectorType m_detector_type;

  string m_ranknet_config_file;

  po::variables_map m_variable_map;

  po::options_description m_visible_options;

  bool m_evolve_model;

  bool m_detect_all_reports;

  bool m_indexing;

  bool m_record_recommendation;

  bool m_analyzing_time_interval;

  unsigned m_time_constraint;

  IndexingType::EnumIndexingType m_indexing_type;

//  string m_validation_error;
  vector<string> m_validation_errors;

  double m_indexing_idf_threshold;

public:

  static void initialize_singleton(int argc, char *argv[]);

  static CmdOption& get_instance();

  static void dispose_singleton();

private:

  CmdOption(int argc, char *argv[]);

  ~CmdOption();

public:

  bool indexing() const;

  bool recording_recommendation() const;

  bool analyzing_time_intervals() const;

  unsigned get_time_constraint() const;

  unsigned indexing_summary_weight() const;

  unsigned indexing_param_top_cosine_number() const;

  double get_indexing_idf_threshold() const;

  unsigned number_of_training_duplicates() const;

  IndexingType::EnumIndexingType indexing_type() const;

  int get_top_k() const;

  string get_ranknet_config_file() const;

  bool detecting_all_reports() const;

  string get_recommendation_file() const;

  PlainSimilarityMeasureFactory::SimilarityMeasureType get_plain_similarity_type() const;

  DetectorFactory::EnumDetectorType get_detector_type() const;

  bool is_using_ranknet_detector() const;

  bool is_using_plain_detector() const;

  bool is_using_svm_detector() const;

  bool is_to_evolve_model() const;

  int get_iteration() const;

  FeatureVectorCalculatorFactory::EnumFeatureCalculatorType get_feature_calculator_type() const;

  const string& get_project_name() const;

  const string& get_dataset_path() const;

  const string& get_timestamp_file() const;

  const string& get_train_file() const;

  bool has_parsing_error() const;

  void print_parsing_error() const;

  bool has_help() const;

  void print_help() const;

  bool has_version() const;

  void print_version() const;

  void validate_options();

  bool has_validation_error() const;

  void print_validation_error() const;

};

#endif /* CMDOPTION_H_ */
