/*
 * DetectorFactory.h
 *
 *  Created on: Jan 10, 2011
 *      Author: Chengnian SUN
 */

#ifndef DETECTORFACTORY_H_
#define DETECTORFACTORY_H_

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../detection-model/IndexingType.h"
#include "../util/MacroUtility.h"

using namespace std;

class AbstractDuplicateDetector;
class IDetectorExtension;
class ReportDataset;

class DetectorFactory {
public:
  enum EnumDetectorType {
    NONE = 0,

    SVM,

    PLAIN,

    RANK_NET,

//    INDEXED_RANK_NET,

  };

  static enum EnumDetectorType parse_detector_type(const int int_type);

  static string get_detector_type_string(const EnumDetectorType type);

  static string get_detector_type_mapping();

  static AbstractDuplicateDetector* create_detector(FILE* log_file,
      const unsigned count_to_skip,
      const vector<IDetectorExtension*>& extensions,
      const ReportDataset& report_dataset);

private:

  DetectorFactory();

  ~DetectorFactory();
};

#endif /* DETECTORFACTORY_H_ */
