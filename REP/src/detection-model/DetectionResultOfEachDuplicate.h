/*
 * DetectionResultOfEachDuplicate.h
 *
 *  Created on: Jan 12, 2011
 *      Author: Chengnian SUN
 */

#ifndef DETECTIONRESULTOFEACHDUPLICATE_H_
#define DETECTIONRESULTOFEACHDUPLICATE_H_

#include <climits>
#include <ctime>

class DetectionResultOfEachDuplicate {
public:
  const static unsigned INVALID_INDEX_OF_MASTER_DETECTED = UINT_MAX;

private:

  unsigned m_number_of_duplicates;

  int m_duplicate_report_id;

  int m_master_report_id;

  unsigned m_index_where_master_detected;

  time_t m_time_cost;

public:

  DetectionResultOfEachDuplicate(unsigned number_of_duplicates,
      int duplicate_report_id, int master_report_id,
      unsigned index_where_master_detected, time_t time_cost);

  ~DetectionResultOfEachDuplicate();

  inline unsigned get_number_of_duplicates() const {
    return this->m_number_of_duplicates;
  }

  inline int get_duplicate_report_id() const {
    return this->m_duplicate_report_id;
  }

  inline int get_master_report_id() const {
    return this->m_master_report_id;
  }

  inline unsigned get_index_where_master_detected() const {
    return this->m_index_where_master_detected;
  }

  inline time_t get_time_cost() const {
    return this->m_time_cost;
  }

};
#endif /* DETECTIONRESULTOFEACHDUPLICATE_H_ */
