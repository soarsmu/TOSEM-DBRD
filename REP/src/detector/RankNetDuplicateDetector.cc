/*
 * RankNetDuplicateDetector.cc
 *
 *  Created on: Jan 5, 2011
 *      Author: Chengnian SUN
 */

#include "../ranknet/DefaultREPParameter.h"
#include "../topping/IndexedRankNetToppingAlgorithm.h"
#include "../topping/RankNetToppingAlgorithm.h"
#include "RankNetDuplicateDetector.h"

void RankNetDuplicateDetector::log_detector_summary(FILE* log_file) {
  fprintf(log_file, "RankNet Detector\n");
}

AbstractToppingAlgorithm* RankNetDuplicateDetector::create_topping_algorithm(
    const ReportBuckets& buckets) {
  if (this->m_using_index) {
    return new IndexedRankNetToppingAlgorithm(this->get_log_file(), buckets,
        *(this->m_default_model_parameter));
  } else {
    return new RankNetToppingAlgorithm(this->get_log_file(), buckets,
        *(this->m_default_model_parameter));
  }
}

RankNetDuplicateDetector::RankNetDuplicateDetector(FILE* log_file,
    unsigned top_number, const ReportDataset& report_dataset,
    const unsigned count_to_skip, const vector<IDetectorExtension*>& extensions,
    const string default_model_parameter_config_file,
    const bool detecting_all_reports,
    const IndexingType::EnumIndexingType indexing_type, const bool using_index)
    : AbstractDuplicateDetector(log_file, top_number, report_dataset,
        count_to_skip, extensions, detecting_all_reports, indexing_type), m_default_model_parameter(
        new DefaultREPParameter(default_model_parameter_config_file)), m_using_index(
        using_index) {
}

RankNetDuplicateDetector::~RankNetDuplicateDetector() {
  delete this->m_default_model_parameter;
}
