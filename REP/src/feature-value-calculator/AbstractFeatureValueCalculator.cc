/*
 * AbstractFeatureValueCalculator.cpp
 *
 *  Created on: Jan 6, 2011
 *      Author: Chengnian SUN
 */

#include "AbstractFeatureValueCalculator.h"

AbstractFeatureValueCalculator::AbstractFeatureValueCalculator(
    const ReportBuckets& report_buckets) :
    m_report_buckets(report_buckets) {

}

AbstractFeatureValueCalculator::~AbstractFeatureValueCalculator() {
}
