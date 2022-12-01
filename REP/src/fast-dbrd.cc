//============================================================================
// Name        : fast-dbrd.cpp
// Author      : Chengnian SUN
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace std;

#include "util/CmdOption.h"
#include "detector/AbstractDuplicateDetector.h"
#include "extensions/IDetectorExtension.h"
#include "extensions/OnlineResultCollector.h"
#include "extensions/RecallAndTimeIntervalAnalysis.h"
#include "extensions/RecommendationRecorder.h"
#include "util/RandomUtility.h"
#include "util/MacroUtility.h"
#include "detector/DetectorFactory.h"
#include "report-model/ReportDataset.h"

using namespace std;

void print_average_result_statistics(FILE* file, const char* name,
    vector<vector<std::pair<double, int> > >& correct_list_list,
    const bool to_print_rank) {
  fprintf(file, "Average %s...\n", name);
  const unsigned number_of_iterations = correct_list_list.size();
  assert(number_of_iterations > 0);
  const unsigned number_of_top = correct_list_list[0].size();
  const unsigned denominator = number_of_iterations;
  for (unsigned rank = 0; rank < number_of_top; rank++) {
    double sum = 0;
    for (unsigned j = 0; j < number_of_iterations; j++) {
      sum += correct_list_list[j][rank].first;
    }
    if (to_print_rank) {
      fprintf(file, "TOP(%2u)=%f\n", (rank + 1),
          (static_cast<double>(sum) / denominator));
    } else {
      fprintf(file, "%f\n", (static_cast<double>(sum) / denominator));
    }
  }
}

void print_result_statistics(FILE* file, const char* name,
    const vector<std::pair<double, int> >& list,
    const unsigned number_of_total_duplicates, const double denominator,
    const bool to_print_rank) {
  fprintf(file, "%s...\n", name);
  const unsigned size_of_top = list.size();

  for (unsigned i = 0; i < size_of_top; i++) {
    const std::pair<double, int>& value = list[i];
    if (to_print_rank) {
      fprintf(file, "TOP(%2u)(%5d/%5u)=%f\n", (i + 1), value.second,
          number_of_total_duplicates, value.first / denominator);
    } else {
      fprintf(file, "%f\n", value.first);
    }
  }
}

void print_file_header(FILE* file, const CmdOption& option) {
  time_t total_start_time = time(NULL);
  fprintf(file,
      "====================================================================\n");
  fprintf(file, "--Started at %s", ctime(&total_start_time));
  fprintf(file, "--Dataset file = %s\n", option.get_dataset_path().c_str());
  fprintf(file, "--Total Iterations = %u\n", option.get_iteration());
  fprintf(file, "--Indexing Type = %s\n",
      IndexingType::get_indexing_type_string(option.indexing_type()).c_str());
  fprintf(file,
      "====================================================================\n\n\n");
}

static std::string get_file_name(string file_prefix, const CmdOption& option) {
  char name[400];
  const char* project_name = option.get_project_name().c_str();
  const int evolution = option.is_to_evolve_model();
  const int iteration = option.get_iteration();

  if (option.is_using_svm_detector()) {
    const char* feature_name =
        FeatureVectorCalculatorFactory::get_enum_feature_calculator_type_string(
            option.get_feature_calculator_type()).c_str();
    sprintf(name, "%s_%s_%s_F-%s_E-%s_I-%d", file_prefix.c_str(), project_name,
        "svm", feature_name, (evolution ? "true" : "false"), iteration);
  } else if (option.is_using_plain_detector()) {
    const char* measure_name =
        PlainSimilarityMeasureFactory::get_similarity_measure_type_string(
            option.get_plain_similarity_type()).c_str();
    sprintf(name, "%s_%s_%s_P-%s_I-%d", file_prefix.c_str(), project_name,
        "plain", measure_name, iteration);
  } else if (option.is_using_ranknet_detector()) {
    sprintf(name, "%s_%s_%s_I-%d", file_prefix.c_str(), "ranknet", project_name,
        iteration);
  } else {
    UNREACHABLE("unrecognized detector type!");
  }
  string result = name;
  return result;
}

FILE* open_time_interval_analysis_file(const CmdOption& option) {
  if (!option.analyzing_time_intervals())
    return NULL;
  else {
    FILE* file = fopen(get_file_name("time-analysis", option).c_str(), "w");
    if (file)
      return file;
    else {
      UNREACHABLE("Cannot open time interval analysis file!");
      return NULL;
    }
  }
}

FILE* open_recommendation_file(const CmdOption& option) {
  if (!option.recording_recommendation()) {
    return NULL;
  } else {
    FILE* file = fopen(get_file_name("recommend", option).c_str(), "w");
    if (file)
      return file;
    else {
      UNREACHABLE("Cannot open recommendation file!");
      return NULL;
    }
  }
}

FILE* open_result_file(const CmdOption& option) {
  const char* name = get_file_name("dbrd", option).c_str();
  FILE* result = fopen(get_file_name("dbrd", option).c_str(), "w");
  if (result == NULL) {
    char message[200];
    sprintf(message, "cannot create output file %s\n", name);
    ERROR_HERE(message);
  }
  return result;
}

void start_detection(const CmdOption& option) {
  vector<vector<std::pair<double, int> > > recall_list;
  vector<double> map_list;
  vector<time_t> times;
  const unsigned total_iterations = option.get_iteration();
  unsigned number_of_duplicates = 0;
  FILE* file = open_result_file(option);
  FILE* recommendation_file = open_recommendation_file(option);
  FILE* time_analysis_file = open_time_interval_analysis_file(option);
  print_file_header(file, option);
  ReportDataset dataset(option.get_dataset_path(), option.get_timestamp_file());
  const int top_k = option.get_top_k();
  for (unsigned iteration = 0; iteration < total_iterations; iteration++) {

    fprintf(file, "Iteration %u\n", iteration + 1);
    fprintf(stdout, "Iteration %u\n", iteration + 1);
    if (recommendation_file)
      fprintf(recommendation_file, "Iteration %u\n", iteration + 1);
    if (time_analysis_file)
      fprintf(time_analysis_file, "Iteration %u\n", iteration + 1);

    const time_t start_time = time(NULL);
    const unsigned seed = RandomUtility::get_default().re_seed();
    fprintf(file, "Random seed = %u\n", seed);

    vector<IDetectorExtension*> extentions;

    OnlineResultCollector* collector = new OnlineResultCollector(top_k, file);
    extentions.push_back(collector);


    RecommendationRecorder* recorder = NULL;
    if (recommendation_file) {
      recorder = new RecommendationRecorder(top_k, recommendation_file);
      extentions.push_back(recorder);
    }
    RecallAndTimeIntervalAnalysis* analysis = NULL;
    if (time_analysis_file) {
      analysis = new RecallAndTimeIntervalAnalysis(top_k, time_analysis_file);
      extentions.push_back(analysis);
    }

    AbstractDuplicateDetector* detector = DetectorFactory::create_detector(file,
        top_k, extentions, dataset);

    detector->init();
    detector->detect();

    const time_t elapsed_time = time(NULL) - start_time;
    times.push_back(elapsed_time);
    fprintf(file, "\nElapsed time = %u\n", static_cast<unsigned>(elapsed_time));
    fprintf(stdout, "\nElapsed time = %u\n",
        static_cast<unsigned>(elapsed_time));

    assert(
        number_of_duplicates == 0 || number_of_duplicates == collector->get_number_of_duplicates());
    number_of_duplicates = collector->get_number_of_duplicates();

    const vector<std::pair<double, int> >& recall =
        collector->get_recall_result();
    recall_list.push_back(recall);
    print_result_statistics(file, "Recall", recall, number_of_duplicates, 1,
        true);
    print_result_statistics(file, "Recall", recall, number_of_duplicates, 1,
        false);

    const double map = collector->get_mean_average_precision();
    map_list.push_back(map);
    fprintf(file, "MAP = %f\n", map);

    detector->dispose();

    delete detector;
    delete collector;
    delete recorder;
    delete analysis;
//    delete recorder;

    fprintf(file, "\n\n\n");
    fprintf(stdout, "\n\n\n");
    if (recommendation_file) {
      fprintf(recommendation_file, "\n\n\n");
      fflush(recommendation_file);
    }
    if (time_analysis_file) {
      fprintf(time_analysis_file, "\n\n\n");
      fflush(time_analysis_file);
    }
    fflush(file);

  }
  print_average_result_statistics(file, "Average Recall", recall_list, true);
  print_average_result_statistics(file, "Average Recall", recall_list, false);
  const double average_map = accumulate(map_list.begin(), map_list.end(), 0.0f)
      / total_iterations;
  fprintf(file, "Average MAP = %f\n", average_map);
  if (recommendation_file)
    fclose(recommendation_file);

  if (time_analysis_file)
    fclose(time_analysis_file);

  fclose(file);
  RandomUtility::dispose_default();
  CmdOption::dispose_singleton();
}

int main(int argc, char* argv[]) {
  CmdOption::initialize_singleton(argc, argv);

  const CmdOption& option = CmdOption::get_instance();
  if (option.has_parsing_error()) {
    option.print_parsing_error();
    CmdOption::dispose_singleton();
    return EXIT_FAILURE;
  } else if (option.has_help()) {
    option.print_help();
    CmdOption::dispose_singleton();
    return EXIT_SUCCESS;
  } else if (option.has_version()) {
    option.print_version();
    CmdOption::dispose_singleton();
    return EXIT_SUCCESS;
  }

  if (option.has_validation_error()) {
    option.print_validation_error();
    option.print_help();
    CmdOption::dispose_singleton();
    return EXIT_FAILURE;
  }

  start_detection(option);
  return EXIT_SUCCESS;
}
