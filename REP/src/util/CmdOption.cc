/*
 * CmdOption.cc
 *
 *  Created on: Jan 13, 2011
 *      Author: Chengnian SUN
 */

#include <climits>
#include <cstdlib>

#include "CmdOption.h"

extern string get_version_string();

#define TO_STRING_HELPER(token) #token
#define TO_STRING(token) TO_STRING_HELPER(token)

#define DEFAULT_TRAINING_REPORTS  200
#define DEFAULT_INDEXING_IDF_THRESHOLD  1.6
#define DEFAULT_INDEXING_PARAM_TOP_COSINE_NUMBER 1200
#define DEFAULT_ITERATION 1
#define DEFAULT_TOP_K 20
#define DEFAULT_INDEXING_SUMMARY_WEIGHT 2

#define OP_INDEXING "indexing"

#define OP_TIMESTAMP "ts"

#define OP_ANALYZE_TIME_INTERVALS "analyze-time-intervals"

#define OP_TIME_CONSTRAINT "time-constraint"

#define OP_INDEXING_SUMMARY_WEIGHT "indexing-summary-weight"
#define OP_INDEXING_IDF_THRESHOLD "indexing-idf-threshold"

#define OP_INDEXING_PARAM_TOP_COSINE_NUMBER "indexing-top-cosine"

#define OP_INDEXING_TYPE "indexing-type"
#define OP_INDEXING_TYPE_TAG "indexing-type"

#define OP_DETECTOR_TYPE "detector"
#define OP_DETECTOR_TYPE_TAG "detector,d"

#define OP_EVOLVE_MODEL "evolve"
#define OP_EVOLVE_MODEL_TAG "evolve,e"

#define OP_DETECT_ALL_REPORTS "all-reports"
#define OP_DETECT_ALL_REPORTS_TAG "all-reports"

#define OP_TRAINING_DUPLICATES "training-duplicates"
#define OP_TRAINING_DUPLICATES_TAG OP_TRAINING_DUPLICATES

#define OP_TOP_K "topk"
#define OP_TOP_K_TAG "topk,k"

#define OP_ITERATION "iteration"
#define OP_ITERATION_TAG "iteration,i"

#define OP_NAME "name"
#define OP_NAME_TAG "name,n"

#define OP_FEATURE_CALCULATOR_TYPE "feature-calculator-type"
#define OP_FEATURE_CALCULATOR_TYPE_TAG "feature-calculator-type,f"

#define OP_PLAIN_MEASURE_TYPE "plain-measure-type"
#define OP_PLAIN_MEASURE_TYPE_TAG "plain-measure-type,p"

#define OP_HELP "help"
#define OP_HELP_TAG "help,h"

#define OP_VERSION "version"
#define OP_VERSION_TAG "version,v"

#define OP_RECOMMENDATION "recommend"

#define OP_DS "ds"
#define OP_DS_TAG "ds"

#define OP_RANKNET_CONFIG "ranknetconfig"
#define OP_RANKNET_CONFIG_TAG "ranknetconfig,r"

#define OP_TRAIN_FILE "trainfile"

CmdOption* CmdOption::INSTANCE = NULL;

bool CmdOption::analyzing_time_intervals() const {
  return this->m_analyzing_time_interval;
}

unsigned CmdOption::get_time_constraint() const {
  return this->m_time_constraint;
}

unsigned CmdOption::indexing_summary_weight() const {
  return this->m_indexing_summary_weight;
}

unsigned CmdOption::indexing_param_top_cosine_number() const {
  return this->m_indexing_top_cosine_number;
}

unsigned CmdOption::number_of_training_duplicates() const {
  return this->m_training_duplicates;
}

IndexingType::EnumIndexingType CmdOption::indexing_type() const {
  return this->m_indexing_type;
}

int CmdOption::get_top_k() const {
  return this->m_top_k;
}

string CmdOption::get_ranknet_config_file() const {
  if (!this->is_using_ranknet_detector()) {
    fprintf(stderr,
        "the detector type is not RankNetDetector, cannot retrieve ranknet config file.");
    exit(EXIT_FAILURE);
  } else {
    return this->m_ranknet_config_file;
  }
}

bool CmdOption::detecting_all_reports() const {
  return this->m_detect_all_reports;
}

//string CmdOption::get_recommendation_file() const {
//  return this->m_recommendation_file;
//}
bool CmdOption::recording_recommendation() const {
  return this->m_record_recommendation;
}

PlainSimilarityMeasureFactory::SimilarityMeasureType CmdOption::get_plain_similarity_type() const {
  return this->m_plain_measure_type;
}

DetectorFactory::EnumDetectorType CmdOption::get_detector_type() const {
  return this->m_detector_type;
}

bool CmdOption::is_using_ranknet_detector() const {
  return this->m_detector_type == DetectorFactory::RANK_NET;
}

bool CmdOption::is_using_plain_detector() const {
  return this->m_detector_type == DetectorFactory::PLAIN;
}

bool CmdOption::is_using_svm_detector() const {
  return this->m_detector_type == DetectorFactory::SVM;
}

double CmdOption::get_indexing_idf_threshold() const {
  return this->m_indexing_idf_threshold;
}

bool CmdOption::is_to_evolve_model() const {
  return this->m_evolve_model;
}

int CmdOption::get_iteration() const {
  return this->m_iteration;
}

FeatureVectorCalculatorFactory::EnumFeatureCalculatorType CmdOption::get_feature_calculator_type() const {
  return this->m_feature_calculator_type;
}

const string& CmdOption::get_project_name() const {
  return this->m_project_name;
}

const string& CmdOption::get_dataset_path() const {
  return this->m_dataset;
}

const string& CmdOption::get_timestamp_file() const {
  return this->m_timestamp_file;
}

const string& CmdOption::get_train_file() const {
  return this->m_train_file;
}

bool CmdOption::has_parsing_error() const {
  return !(this->m_parsing_error.empty());
}

void CmdOption::print_parsing_error() const {
  cout << this->m_parsing_error << '\n';
}

void CmdOption::initialize_singleton(int argc, char *argv[]) {
  if (INSTANCE) {
    delete INSTANCE;
    INSTANCE = NULL;
  }
  assert(!INSTANCE);
  INSTANCE = new CmdOption(argc, argv);
  INSTANCE->validate_options();
}

CmdOption& CmdOption::get_instance() {
  assert(INSTANCE);
  return *INSTANCE;
}

void CmdOption::dispose_singleton() {
  assert(INSTANCE);
  delete INSTANCE;
  INSTANCE = NULL;
}

bool CmdOption::has_help() const {
  return this->m_variable_map.count(OP_HELP);
}

void CmdOption::print_help() const {
  cout << this->m_visible_options << '\n';
}

bool CmdOption::has_version() const {
  return this->m_variable_map.count(OP_VERSION);
}

void CmdOption::print_version() const {
  cout << get_version_string() << std::endl;
}

bool CmdOption::has_validation_error() const {
  return this->m_validation_errors.size();
}

void CmdOption::print_validation_error() const {
  for (vector<string>::const_iterator iter = this->m_validation_errors.begin(),
      end = this->m_validation_errors.end(); iter != end; ++iter) {
    cout << "Option Error: " << *iter << "\n";
  }
}

void CmdOption::validate_options() {

  this->m_validation_errors.clear();

  /**
   * summary weight in term weighting for indexing.
   */
  if (this->m_variable_map.count(OP_INDEXING_SUMMARY_WEIGHT)) {
    const int number =
        this->m_variable_map[OP_INDEXING_SUMMARY_WEIGHT].as<int>();
    if (number <= 0) {
      this->m_validation_errors.push_back("Option: "
      OP_INDEXING_SUMMARY_WEIGHT
      " should be greater than 0");
    } else {
      this->m_indexing_summary_weight = number;
    }
  }

  /**
   * number of training duplicate reports.
   */
  if (this->m_variable_map.count(OP_TRAINING_DUPLICATES)) {
    const int number = this->m_variable_map[OP_TRAINING_DUPLICATES].as<int>();
    if (number < 0) {
      this->m_validation_errors.push_back("Option "
      OP_TRAINING_DUPLICATES
      " should be greater than or equal to 0");
    } else {
      this->m_training_duplicates = number;
    }
  } else {
    this->m_training_duplicates = DEFAULT_TRAINING_REPORTS;
  }

  /**
   * using index
   */
  this->m_indexing = this->m_variable_map.count(OP_INDEXING);

  /**
   * time stamp
   */
  if (this->m_variable_map.count(OP_TIMESTAMP)) {
    this->m_timestamp_file = this->m_variable_map[OP_TIMESTAMP].as<string>();
  }

   /**
   * training file
   */
  if (this->m_variable_map.count(OP_TRAIN_FILE)) {
    this->m_train_file = this->m_variable_map[OP_TRAIN_FILE].as<string>();
  }

  /**
   * top k
   */
  if (this->m_variable_map.count(OP_TOP_K)) {
    const int k = this->m_variable_map[OP_TOP_K].as<int>();
    if (k <= 0) {
      this->m_validation_errors.push_back("Option "
      OP_TOP_K
      " should be greater than 0");
    } else {
      this->m_top_k = k;
    }
  }

  /*
   * iteration count
   */
  if (!(this->m_variable_map.count(OP_ITERATION))) {
    this->m_iteration = DEFAULT_ITERATION;
  } else {
    this->m_iteration = this->m_variable_map[OP_ITERATION].as<int>();
    if (this->m_iteration < 1) {
      this->m_validation_errors.push_back("Option "
      OP_ITERATION
      " should be an integer greater than 0");
    }
  }

  /*
   * dataset path
   */
  if (!(this->m_variable_map.count(OP_DS))) {
    this->m_validation_errors.push_back("Dataset path is not specified");
  } else {
    this->m_dataset = this->m_variable_map[OP_DS].as<string>();
  }

  /*
   * project name
   */
  if (!(this->m_variable_map.count(OP_NAME))) {
    this->m_validation_errors.push_back("Project name is not specified");
  } else {
    this->m_project_name = this->m_variable_map[OP_NAME].as<string>();
  }

  /*
   * whether to record recommendation
   */
//  if (!(this->m_variable_map.count(OP_RECOMMENDATION)))
//    this->m_recommendation_file = "";
//  else
//    this->m_recommendation_file = this->m_variable_map[OP_RECOMMENDATION].as<
//        string>();
  this->m_record_recommendation = this->m_variable_map.count(OP_RECOMMENDATION);

  /**
   * analyze time intervals of duplicate reports
   */
  this->m_analyzing_time_interval = this->m_variable_map.count(
      OP_ANALYZE_TIME_INTERVALS);

  /**
   * constraint time
   */
//  this->m_constraining_time = this->m_variable_map.count(OP_TIME_CONSTRAINT);
  if (this->m_variable_map.count(OP_TIME_CONSTRAINT)) {
    this->m_time_constraint = this->m_variable_map[OP_TIME_CONSTRAINT].as<
        unsigned>();
  } else {
    this->m_time_constraint = UINT_MAX;
  }

  /*
   * whether to detect all reports, default, it is false
   */
  if (!(this->m_variable_map.count(OP_DETECT_ALL_REPORTS)))
    this->m_detect_all_reports = false;
  else {
    this->m_detect_all_reports = true;
  }

  /**
   * top cosine number
   */
  if (this->m_variable_map.count(OP_INDEXING_PARAM_TOP_COSINE_NUMBER)) {
    const unsigned number =
        this->m_variable_map[OP_INDEXING_PARAM_TOP_COSINE_NUMBER].as<unsigned>();
    this->m_indexing_top_cosine_number = number;
  } else {
    this->m_indexing_top_cosine_number =
        DEFAULT_INDEXING_PARAM_TOP_COSINE_NUMBER;
  }

  /**
   * idf threshold used in index elimination
   */
  if (this->m_variable_map.count(OP_INDEXING_IDF_THRESHOLD)) {
    const double threshold = this->m_variable_map[OP_INDEXING_IDF_THRESHOLD].as<
        double>();
    if (threshold < 0) {
      this->m_validation_errors.push_back("Option "
      OP_INDEXING_IDF_THRESHOLD
      ": negative IDF threshold is invalid.");
    } else {
      this->m_indexing_idf_threshold = threshold;
    }
  } else {
    this->m_indexing_idf_threshold = DEFAULT_INDEXING_IDF_THRESHOLD;
  }

  /*
   * indexing type
   */
  if (!(this->m_variable_map.count(OP_INDEXING_TYPE))) {
    this->m_indexing_type = IndexingType::NO_INDEXING;
  } else {
    this->m_indexing_type = IndexingType::parse_indexing_type(
        this->m_variable_map[OP_INDEXING_TYPE].as<int>());
  }

#ifndef ONLY_ENABLE_RANKNET
  /*
   * detector type
   */
  if (!(this->m_variable_map.count(OP_DETECTOR_TYPE))) {
    this->m_validation_errors.push_back(
        "Duplicate detector type is not specified");
    return;
  } else {
    this->m_detector_type = DetectorFactory::parse_detector_type(
        this->m_variable_map[OP_DETECTOR_TYPE].as<int>());

    switch (this->m_detector_type) {
      case DetectorFactory::SVM: {
        // SVM DETECTOR, e.g. ICSE10

        /*
         * feature vector calculator
         */
        if (!(this->m_variable_map.count(OP_FEATURE_CALCULATOR_TYPE))) {
          this->m_validation_errors.push_back("Option "
              OP_FEATURE_CALCULATOR_TYPE
              " is not specified");
        } else {
          int num_feature_calculator_type =
          this->m_variable_map[OP_FEATURE_CALCULATOR_TYPE].as<int>();
          if (num_feature_calculator_type < 1) {
            this->m_validation_errors.push_back("Option "
                OP_FEATURE_CALCULATOR_TYPE
                " should be an integer greater than 0");
          }
          this->m_feature_calculator_type =
          FeatureVectorCalculatorFactory::get_enum_feature_calculator_type_from_int(
              num_feature_calculator_type);
        }

        /*
         * evolve model duing detection?
         */
        if (!(this->m_variable_map.count(OP_EVOLVE_MODEL))) {
          this->m_validation_errors.push_back(
              "Evloving model flag is not specified");
        } else {
          this->m_evolve_model = this->m_variable_map[OP_EVOLVE_MODEL].as<bool>();
        }
        break;
      }
      case DetectorFactory::PLAIN: {
        // PLAIN DETECTOR.
        if (!(this->m_variable_map.count(OP_PLAIN_MEASURE_TYPE))) {
          this->m_validation_errors.push_back(
              "Plain similarity measure type is not specified");
        } else {
          this->m_plain_measure_type =
          PlainSimilarityMeasureFactory::parse_similarity_measure_type(
              this->m_variable_map[OP_PLAIN_MEASURE_TYPE].as<int>());
        }
        break;
      }
      case DetectorFactory::RANK_NET: {
        // ranknet detector
        if (!(this->m_variable_map.count(OP_RANKNET_CONFIG))) {
          this->m_validation_errors.push_back(
              "RankNet config file path is not specified");
        } else {
          this->m_ranknet_config_file =
          this->m_variable_map[OP_RANKNET_CONFIG].as<string>();
        }
        break;
      }
      default:
      this->m_validation_errors.push_back("Unacceptable detector type");
      break;
    }
  }
#else
  this->m_detector_type = DetectorFactory::RANK_NET;
  // ranknet detector
  if (!(this->m_variable_map.count(OP_RANKNET_CONFIG))) {
    this->m_validation_errors.push_back(
        "RankNet config file path is not specified");
  } else {
    this->m_ranknet_config_file = this->m_variable_map[OP_RANKNET_CONFIG].as<
        string>();
  }
#endif
}

CmdOption::~CmdOption() {
}

static void build_necessary_options(
    po::options_description& necessary_options) {
  necessary_options.add_options()(OP_NAME_TAG, po::value<string>(),
      "project name");

  necessary_options.add_options()(OP_RANKNET_CONFIG_TAG, po::value<string>(),
      "ranknet model config file.");

  necessary_options.add_options()(OP_TRAIN_FILE, po::value<string>(), "train file");

#ifndef ONLY_ENABLE_RANKNET

  necessary_options.add_options()(OP_FEATURE_CALCULATOR_TYPE_TAG,
      po::value<int>(),
      ("feature calculator type. "
          + FeatureVectorCalculatorFactory::get_enum_feature_calculator_type_mapping()).c_str());

  necessary_options.add_options()(OP_PLAIN_MEASURE_TYPE_TAG, po::value<int>(),
      ("plain measure type. "
          + PlainSimilarityMeasureFactory::get_similarity_measure_type_mapping()).c_str());

  necessary_options.add_options()(OP_DETECTOR_TYPE_TAG, po::value<int>(),
      DetectorFactory::get_detector_type_mapping().c_str());

  necessary_options.add_options()(OP_EVOLVE_MODEL_TAG, po::value<bool>(),
      "whether to evlove the model");
#endif
}

bool CmdOption::indexing() const {
  return this->m_indexing;
}

static void build_timing_options(po::options_description& options) {
//  timing_options.add_options()("ts,t", po::value<string>(),
//      "the file storing timestamps for bug reports");

  options.add_options()(OP_TIMESTAMP, po::value<string>(),
      "the file storing timestamps for bug reports");

  options.add_options()(OP_ANALYZE_TIME_INTERVALS,
      "analyze time intervals of duplicate reports");

  options.add_options()(OP_TIME_CONSTRAINT, po::value<unsigned>(),
      "confine the search scope within a time range");
}

static void build_indexing_options(po::options_description& indexing_options) {

  indexing_options.add_options()(OP_INDEXING,
      "whether to use index for instance retrieval");

  indexing_options.add_options()(OP_INDEXING_TYPE_TAG, po::value<int>(),
      IndexingType::get_indexing_type_mapping().c_str());

  indexing_options.add_options()(OP_INDEXING_PARAM_TOP_COSINE_NUMBER,
      po::value<unsigned>(),
      "the number of reports with the top cosine similarity "
          "retained in indexing retrieval."
          "The default value is "
      TO_STRING(DEFAULT_INDEXING_PARAM_TOP_COSINE_NUMBER));

  indexing_options.add_options()(OP_INDEXING_IDF_THRESHOLD, po::value<double>(),
      "IDF threshold in index elimination. The default value is "
      TO_STRING(DEFAULT_INDEXING_IDF_THRESHOLD));

  indexing_options.add_options()(OP_INDEXING_SUMMARY_WEIGHT, po::value<int>(),
      "summary weight for term weighting in indexing. "
          "The default value is "
      TO_STRING(DEFAULT_INDEXING_SUMMARY_WEIGHT));
}

static void build_other_options(po::options_description& options) {
  options.add_options()(OP_TRAINING_DUPLICATES_TAG, po::value<int>(),
      "number of duplicates to construct the training set. The default value is "
      TO_STRING(DEFAULT_TRAINING_REPORTS));

  options.add_options()(OP_TOP_K_TAG, po::value<int>(),
      "number of reports to retrieve for each new report. "
          "The default value is "
      TO_STRING(DEFAULT_TOP_K));

  options.add_options()(OP_ITERATION_TAG, po::value<int>(),
      "number of iterations. The default value is "
      TO_STRING(DEFAULT_ITERATION));

  options.add_options()(OP_RECOMMENDATION, "record recommendation");

  options.add_options()(OP_DETECT_ALL_REPORTS_TAG,
      "whether to detect all reports");
}

CmdOption::CmdOption(int argc, char *argv[]) {
  this->m_training_duplicates = DEFAULT_TRAINING_REPORTS;
  this->m_iteration = DEFAULT_ITERATION;
  this->m_feature_calculator_type = FeatureVectorCalculatorFactory::NONE;
  this->m_plain_measure_type = PlainSimilarityMeasureFactory::NONE;
  this->m_detector_type = DetectorFactory::NONE;
  this->m_evolve_model = false;
  this->m_project_name = "";
  this->m_dataset = "";
  this->m_detect_all_reports = false;
  this->m_top_k = DEFAULT_TOP_K;
  this->m_indexing_type = IndexingType::NO_INDEXING;
  this->m_indexing_idf_threshold = DEFAULT_INDEXING_IDF_THRESHOLD;
  this->m_indexing_top_cosine_number = DEFAULT_INDEXING_PARAM_TOP_COSINE_NUMBER;
  this->m_indexing_summary_weight = DEFAULT_INDEXING_SUMMARY_WEIGHT;
  this->m_indexing = false;
  this->m_record_recommendation = false;
  this->m_timestamp_file = "";
  this->m_analyzing_time_interval = false;
  this->m_time_constraint = UINT_MAX;
  this->m_train_file = "";
  /**
   * generic options.
   *
   * --version
   *
   * --help
   *
   */
  po::options_description generic_options("Generic Options");
  generic_options.add_options()(OP_VERSION_TAG, "print version string")(
      OP_HELP_TAG, "produce help message");

  /**
   * necessary options:
   *
   * --n=<negative-min-support>
   *
   * --k=<top-k>
   */
  po::options_description necessary_options("Necessary options");
  build_necessary_options(necessary_options);

  po::options_description indexing_options("Indexing options");
  build_indexing_options(indexing_options);

  po::options_description timing_options("Timing options");
  build_timing_options(timing_options);

  po::options_description other_options("Other options");
  build_other_options(other_options);

  po::options_description hidden_options("Hidden options");
  hidden_options.add_options()(OP_DS_TAG, po::value<string>(), "dataset file");

  po::options_description cmdline_options;
  cmdline_options.add(generic_options);
  cmdline_options.add(necessary_options);
  cmdline_options.add(timing_options);
  cmdline_options.add(indexing_options);
  cmdline_options.add(other_options);
  cmdline_options.add(hidden_options);

  this->m_visible_options.add(necessary_options);
  this->m_visible_options.add(timing_options);
  this->m_visible_options.add(indexing_options);
  this->m_visible_options.add(other_options);
  this->m_visible_options.add(generic_options);

  po::positional_options_description dataset_option;
  dataset_option.add(OP_DS_TAG, 1);

  try {
    po::store(
        po::command_line_parser(argc, argv).options(cmdline_options).positional(
            dataset_option).run(), m_variable_map);
    po::notify(m_variable_map);
  } catch (std::exception& e) {
    this->m_parsing_error += e.what();
  }
}
