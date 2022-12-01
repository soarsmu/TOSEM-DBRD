/*
 * DetectorFactory.cc
 *
 *  Created on: Jan 10, 2011
 *      Author: Chengnian SUN
 */
#include <cstdio>

#include "DetectorFactory.h"
#include "../util/CmdOption.h"
#include "../util/MacroUtility.h"

#include "RankNetDuplicateDetector.h"
#include <string>
using namespace std;

enum DetectorFactory::EnumDetectorType DetectorFactory::parse_detector_type(
    const int int_type) {
  enum EnumDetectorType type = static_cast<enum EnumDetectorType>(int_type);
  switch (type) {
  case SVM:
  case PLAIN:
  case RANK_NET:
    return type;
  default: {
    char message[100];
    sprintf(message, "unhandled detector type %d", int_type);
    ERROR_HERE(message);
    return NONE;
  }
  }
}

string DetectorFactory::get_detector_type_string(const EnumDetectorType type) {
  switch (type) {
  case SVM: {
    return "SVM";
  }
  case PLAIN: {
    return "PLAIN";
  }
  case RANK_NET: {
    return "RANK_NET";
  }
//  case INDEXED_RANK_NET: {
//    return "INDEXED_RANK_NET";
//  }
  default: {
    char message[100];
    sprintf(message, "unhandled detector type %d", type);
    ERROR_HERE(message);
    return "";
  }
  }
}

string DetectorFactory::get_detector_type_mapping() {
  stringstream ss;
  ss << SVM << ":SVM, ";
  ss << PLAIN << ":PLAIN, ";
  ss << RANK_NET << ":RANK_NET, ";
//  ss << INDEXED_RANK_NET << ":INDEXED_RANK_NET";
  return ss.str();
}

AbstractDuplicateDetector* DetectorFactory::create_detector(FILE* log_file,
    const unsigned top_number, const vector<IDetectorExtension*>& extensions,
    const ReportDataset& report_dataset) {
  if (log_file == NULL) {
    ERROR_HERE("invalid log file.");
  }
  const CmdOption& option = CmdOption::get_instance();
  const unsigned count_to_skip = option.number_of_training_duplicates();
  const bool detecting_all_reports = option.detecting_all_reports();
  const IndexingType::EnumIndexingType indexing_type = option.indexing_type();
  switch (option.get_detector_type()) {
  //XXX disable all detectors except RANK_NET.
  case SVM:
  case PLAIN:
  case RANK_NET:
//  case INDEXED_RANK_NET:
    return new RankNetDuplicateDetector(log_file, top_number, report_dataset,
        count_to_skip, extensions, option.get_ranknet_config_file(),
        detecting_all_reports, indexing_type,
        option.indexing());
  default:
    char message[100];
    sprintf(message, "unhandled detector type %d", option.get_detector_type());
    ERROR_HERE(message);
    return NULL;
  }
}

DetectorFactory::DetectorFactory() {

}

DetectorFactory::~DetectorFactory() {
}
