/*
 * RankNetDuplicateDetector.h
 *
 *  Created on: Jan 5, 2011
 *      Author: Chengnian SUN
 */

#ifndef RANKNETDUPLICATEDETECTOR_H_
#define RANKNETDUPLICATEDETECTOR_H_

#include "AbstractDuplicateDetector.h"

class DefaultREPParameter;
#include <string>
using namespace std;
class RankNetDuplicateDetector: public AbstractDuplicateDetector {
private:

  const DefaultREPParameter* m_default_model_parameter;

  const bool m_using_index;

protected:

  virtual void log_detector_summary(FILE* log_file);

  virtual AbstractToppingAlgorithm* create_topping_algorithm(
      const ReportBuckets& buckets);

public:

  RankNetDuplicateDetector(FILE* file, unsigned top_number,
      const ReportDataset& report_dataset, const unsigned count_to_skip,
      const vector<IDetectorExtension*>& extensions,
      const string default_model_parameter_config_file,
      const bool detecting_all_reports,
      const IndexingType::EnumIndexingType indexing_type,
      const bool using_index);

  virtual ~RankNetDuplicateDetector();

};

#endif /* RANKNETDUPLICATEDETECTOR_H_ */
