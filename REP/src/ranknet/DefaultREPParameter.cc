/*
 * DefaultModelParameter.cc
 *
 *  Created on: Apr 27, 2011
 *      Author: Chengnian SUN
 */

#include "DefaultREPParameter.h"

#include <iostream>
#include <fstream>
using namespace std;

#include "../config-reader/ConfigFile.h"
#include "../util/MacroUtility.h"
#include <cstdio>
#include <cstdlib>
using namespace std;
#include <string>

#define READ(config, holder, tag) \
	if (!(config).readInto((holder), tag)) { \
		fprintf(stderr, "attribute [%s] does not exist in the config file.", (tag)); \
		exit(EXIT_FAILURE); \
	}

DefaultREPParameter::DefaultREPParameter(const string& config_file_path) {
  try {
    ConfigFile config(config_file_path);
    READ(config, this->m_unigram_weight, "UNIGRAM_WEIGHT");
    READ(config, this->m_unigram_weight_fixed, "UNIGRAM_WEIGHT_FIXED");

    READ(config, this->m_bigram_weight, "BIGRAM_WEIGHT");
    READ(config, this->m_bigram_weight_fixed, "BIGRAM_WEIGHT_FIXED");

    READ(config, this->m_k1, "K1");
    READ(config, this->m_k1_fixed, "K1_FIXED");

    READ(config, this->m_summary_weight, "SUMMARY_WEIGHT");
    READ(config, this->m_summary_weight_fixed, "SUMMARY_WEIGHT_FIXED");

    READ(config, this->m_summary_b, "SUMMARY_B");
    READ(config, this->m_summary_b_fixed, "SUMMARY_B_FIXED");

    READ(config, this->m_description_weight, "DESCRIPTION_WEIGHT");
    READ(config, this->m_description_weight_fixed, "DESCRIPTION_WEIGHT_FIXED");

    READ(config, this->m_description_b, "DESCRIPTION_B");
    READ(config, this->m_description_b_fixed, "DESCRIPTION_B_FIXED");

    READ(config, this->m_component_weight, "COMPONENT_WEIGHT");
    READ(config, this->m_component_weight_fixed, "COMPONENT_WEIGHT_FIXED");

    READ(config, this->m_sub_component_weight, "SUB_COMPONENT_WEIGHT");
    READ(config, this->m_sub_component_weight_fixed,
        "SUB_COMPONENT_WEIGHT_FIXED");

    READ(config, this->m_report_type_weight, "REPORT_TYPE_WEIGHT");
    READ(config, this->m_report_type_weight_fixed, "REPORT_TYPE_WEIGHT_FIXED");

    READ(config, this->m_priority_weight, "PRIORITY_WEIGHT");
    READ(config, this->m_priority_weight_fixed, "PRIORITY_WEIGHT_FIXED");

    READ(config, this->m_version_weight, "VERSION_WEIGHT");
    READ(config, this->m_version_weight_fixed, "VERSION_WEIGHT_FIXED");

    READ(config, this->m_k3, "K3");
    READ(config, this->m_k3_fixed, "K3_FIXED");

    READ(config, this->m_count_of_irrelevant_reports_per_query,
        "COUNT_OF_IRRELEVANT_REPORTS_PER_QUERY");
    READ(config, this->m_max_query_count, "MAX_QUERY_COUNT");

  } catch (ConfigFile::file_not_found& e) {
    fprintf(stderr, "RankNet config file [%s] does not exist!",
        config_file_path.c_str());
    exit(EXIT_FAILURE);
  }
}
