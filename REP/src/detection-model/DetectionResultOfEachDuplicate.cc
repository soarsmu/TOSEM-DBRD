/*
 * DetectionResultOfEachDuplicate.cc
 *
 *  Created on: Jan 12, 2011
 *      Author: Chengnian SUN
 */

#include "DetectionResultOfEachDuplicate.h"

DetectionResultOfEachDuplicate::DetectionResultOfEachDuplicate(
    unsigned number_of_duplicates, int duplicate_report_id,
    int master_report_id, unsigned index_where_master_detected,
    time_t time_cost) :
    m_number_of_duplicates(number_of_duplicates), m_duplicate_report_id(
        duplicate_report_id), m_master_report_id(master_report_id), m_index_where_master_detected(
        index_where_master_detected), m_time_cost(time_cost) {
}

DetectionResultOfEachDuplicate::~DetectionResultOfEachDuplicate() {
}
