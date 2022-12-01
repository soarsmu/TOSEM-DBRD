/*
 * ReportReader.cpp
 *
 *  Created on: 2010-7-29
 *      Author: Chengnian Sun.
 */

#include <cstdlib>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

#include "DuplicateBugReport.h"
#include "MasterBugReport.h"
#include "ReportReader.h"
#include "TimestampMap.h"

using namespace std;

#define TAG_ID "ID="
#define LENGTH_TAG_ID 3

//---Summary Section---
#define TAG_SUMMARY_UNIGRAM "S-U="
#define LENGTH_TAG_SUMMARY_UNIGRAM 4

#define TAG_SUMMARY_BIGRAM "S-B="
#define LENGTH_TAG_SUMMARY_BIGRAM 4

#define TAG_SUMMARY_TRIGRAM "S-T="
#define LENGTH_TAG_SUMMARY_TRIGRAM 4

//---Description Section---
#define TAG_DESCRIPTION_UNIGRAM "D-U="
#define LENGTH_TAG_DESCRIPTION_UNIGRAM 4

#define TAG_DESCRIPTION_BIGRAM "D-B="
#define LENGTH_TAG_DESCRIPTION_BIGRAM 4

#define TAG_DESCRIPTION_TRIGRAM "D-T="
#define LENGTH_TAG_DESCRIPTION_TRIGRAM 4

//---All Section---
#define TAG_ALL_UNIGRAM "A-U="
#define LENGTH_TAG_ALL_UNIGRAM 4

#define TAG_ALL_BIGRAM "A-B="
#define LENGTH_TAG_ALL_BIGRAM 4

#define TAG_ALL_TRIGRAM "A-T="
#define LENGTH_TAG_ALL_TRIGRAM 4

#define TAG_DUPLICATE_ID "DID="
#define LENGTH_TAG_DUPLICATE_ID 4

// ---
#define TAG_VERSION "VERSION="
#define LENGTH_TAG_VERSION 8

#define TAG_COMPONENT "COMPONENT="
#define LENGTH_TAG_COMPONENT 10

#define TAG_SUB_COMPONENT "SUB-COMPONENT="
#define LENGTH_TAG_SUB_COMPONENT 14

#define TAG_TYPE "TYPE="
#define LENGTH_TAG_TYPE 5

#define TAG_PRIORITY "PRIORITY="
#define LENGTH_TAG_PRIORITY 9

#define DELIMITER_CHAR ','

#define ID_FREQUENCY_SEPARATOR ':'

/**
 * @return the max term id;
 */
static inline int tokenize(const string& line,
    const string::size_type start_position, vector<Term>& vec) {
  string::size_type start_index = start_position;
  string::size_type end_index;
  string temp_string;
  int max_term_id = 0;
  while (start_index != string::npos) {
    end_index = line.find_first_of(ID_FREQUENCY_SEPARATOR, start_index);
    if (end_index == string::npos) {
      break;
    }
    temp_string = line.substr(start_index, end_index);
    assert(temp_string.length() > 0);
    int term_id = atoi(temp_string.c_str());
    if (term_id > max_term_id) {
      max_term_id = term_id;
    }

    start_index = end_index + 1;
    end_index = line.find_first_of(DELIMITER_CHAR, start_index);

    temp_string = line.substr(start_index, end_index);
    assert(temp_string.length() > 0);
    int term_frequency = atoi(temp_string.c_str());

    vec.push_back(Term(term_id, term_frequency));

    if (end_index != string::npos) {
      start_index = end_index + 1;
    } else {
      start_index = end_index;
    }
  }
  return max_term_id;
}

inline void temp_get_line(ifstream& file, string& line) {
  getline(file, line);
  const unsigned size = line.size();
  if (size > 0 && line[size - 1] == '\r') {
    line.erase(size - 1);
  }
}

void ReportReader::read_reports_from_file(const string& file_path,
    const string& timestamp_file, vector<AbstractBugReport*>* result_vector) {
  assert(result_vector->empty());

  TimestampMap timestamp_map(timestamp_file);

  ifstream file;
  file.open(file_path.c_str());
  if (!file.is_open()) {
    fprintf(stderr, "ERROR: Cannot open report file %s\n", file_path.c_str());
    exit(1);
  }

  string line;
  vector<Term> summary_unigrams;
  vector<Term> summary_bigrams;
  vector<Term> summary_trigrams;

  vector<Term> description_unigrams;
  vector<Term> description_bigrams;
  vector<Term> description_trigrams;

  vector<Term> all_unigrams;
  vector<Term> all_bigrams;
  vector<Term> all_trigrams;

  int id;
  int duplicate_id;
  int max_term_id = 0;
  int temp_max_term_id;

  // --
  int version;
  int component;
  int sub_component;
  int type;
  int priority;

  while (!file.eof()) {

    // read id
    temp_get_line(file, line);
    if (line.size() == 0) {
      break;
    }
    assert(line.find(TAG_ID) != string::npos);
    id = atoi(line.substr(LENGTH_TAG_ID).c_str());

    // read summary unigrams
    temp_get_line(file, line);
    assert(line.find(TAG_SUMMARY_UNIGRAM) != string::npos);

    temp_max_term_id = tokenize(line, LENGTH_TAG_SUMMARY_UNIGRAM,
        summary_unigrams);
    if (temp_max_term_id > max_term_id) {
      max_term_id = temp_max_term_id;
    }

    // read summary bigrams;
    temp_get_line(file, line);
    assert(line.find(TAG_SUMMARY_BIGRAM) != string::npos);
    temp_max_term_id = tokenize(line, LENGTH_TAG_SUMMARY_BIGRAM,
        summary_bigrams);
    if (temp_max_term_id > max_term_id) {
      max_term_id = temp_max_term_id;
    }

    // read summary trigrams;
    temp_get_line(file, line);
    assert(line.find(TAG_SUMMARY_TRIGRAM) != string::npos);
    temp_max_term_id = tokenize(line, LENGTH_TAG_SUMMARY_TRIGRAM,
        summary_trigrams);
    if (temp_max_term_id > max_term_id) {
      max_term_id = temp_max_term_id;
    }

    // read description unigrams;
    temp_get_line(file, line);
    assert(line.find(TAG_DESCRIPTION_UNIGRAM) != string::npos);
    temp_max_term_id = tokenize(line, LENGTH_TAG_DESCRIPTION_UNIGRAM,
        description_unigrams);
    if (temp_max_term_id > max_term_id) {
      max_term_id = temp_max_term_id;
    }
    // read description bigrams;
    temp_get_line(file, line);
#ifndef NDEBUG
    if (line.find(TAG_DESCRIPTION_BIGRAM) == string::npos) {
      cout << id << endl;
      cout << line << endl << endl;
    }
#endif
    assert(line.find(TAG_DESCRIPTION_BIGRAM) != string::npos);
    temp_max_term_id = tokenize(line, LENGTH_TAG_DESCRIPTION_BIGRAM,
        description_bigrams);
    if (temp_max_term_id > max_term_id) {
      max_term_id = temp_max_term_id;
    }

    // read description trigrams;
    temp_get_line(file, line);
#ifndef NDEBUG
    if (line.find(TAG_DESCRIPTION_TRIGRAM) == string::npos) {
      cout << id << endl;
      cout << line << endl;
    }
#endif
    assert(line.find(TAG_DESCRIPTION_TRIGRAM) != string::npos);
    temp_max_term_id = tokenize(line, LENGTH_TAG_DESCRIPTION_TRIGRAM,
        description_trigrams);
    if (temp_max_term_id > max_term_id) {
      max_term_id = temp_max_term_id;
    }

    // read all unigrams
    temp_get_line(file, line);
    assert(line.find(TAG_ALL_UNIGRAM) != string::npos);
    temp_max_term_id = tokenize(line, LENGTH_TAG_ALL_UNIGRAM, all_unigrams);
    if (temp_max_term_id > max_term_id) {
      max_term_id = temp_max_term_id;
    }

    // read all bigrams
    temp_get_line(file, line);
    assert(line.find(TAG_ALL_BIGRAM) != string::npos);
    temp_max_term_id = tokenize(line, LENGTH_TAG_ALL_BIGRAM, all_bigrams);
    if (temp_max_term_id > max_term_id) {
      max_term_id = temp_max_term_id;
    }

    // read all trigrams;
    temp_get_line(file, line);
    assert(line.find(TAG_ALL_TRIGRAM) != string::npos);
    temp_max_term_id = tokenize(line, LENGTH_TAG_ALL_TRIGRAM, all_trigrams);
    if (temp_max_term_id > max_term_id) {
      max_term_id = temp_max_term_id;
    }

    // read duplicate id
    temp_get_line(file, line);
    assert(line.find(TAG_DUPLICATE_ID) != string::npos);
    string str_did = line.substr(LENGTH_TAG_DUPLICATE_ID);
    if (str_did.length() == 0) {
      duplicate_id = id;
    } else {
      duplicate_id = atoi(str_did.c_str());
    }

    // read version
    temp_get_line(file, line);
    assert(line.find(TAG_VERSION) != string::npos);
    version = atoi(line.substr(LENGTH_TAG_VERSION).c_str());

    // read component
    temp_get_line(file, line);
    assert(line.find(TAG_COMPONENT) != string::npos);
    component = atoi(line.substr(LENGTH_TAG_COMPONENT).c_str());

    // read sub-component
    temp_get_line(file, line);
    assert(line.find(TAG_SUB_COMPONENT) != string::npos);
    sub_component = atoi(line.substr(LENGTH_TAG_SUB_COMPONENT).c_str());

    // read report type;
    temp_get_line(file, line);
    assert(line.find(TAG_TYPE) != string::npos);
    type = atoi(line.substr(LENGTH_TAG_TYPE).c_str());

    // read priority;
    temp_get_line(file, line);
    assert(line.find(TAG_PRIORITY) != string::npos);
    priority = atoi(line.substr(LENGTH_TAG_PRIORITY).c_str());

    AbstractBugReport* report;
    const unsigned timestamp_in_days = timestamp_map.get_timestamp(id);
    if (id == duplicate_id) {
      report = new MasterBugReport(id, summary_unigrams, summary_bigrams,
          summary_trigrams, description_unigrams, description_bigrams,
          description_trigrams, all_unigrams, all_bigrams, all_trigrams,
          version, component, sub_component, type, priority, timestamp_in_days);
    } else {
      report = new DuplicateBugReport(id, duplicate_id, summary_unigrams,
          summary_bigrams, summary_trigrams, description_unigrams,
          description_bigrams, description_trigrams, all_unigrams, all_bigrams,
          all_trigrams, version, component, sub_component, type, priority,
          timestamp_in_days);
    }

    //		BugReport* report = new BugReport(id, duplicate_id, summary_unigrams, summary_bigrams, description_unigrams, description_bigrams,
    //				all_unigrams, all_bigrams);
    result_vector->push_back(report);

    summary_unigrams.clear();
    summary_bigrams.clear();
    summary_trigrams.clear();

    description_unigrams.clear();
    description_bigrams.clear();
    description_trigrams.clear();

    all_unigrams.clear();
    all_bigrams.clear();
    all_trigrams.clear();
  }

  this->max_term_id = max_term_id;
  file.close();
}

int ReportReader::get_max_term_id() const {
  return this->max_term_id;
}

ReportReader::ReportReader(const string& file_path,
    const string& timestamp_file,
    vector<AbstractBugReport*>* report_collector) {
  read_reports_from_file(file_path, timestamp_file, report_collector);
}

ReportReader::~ReportReader() {
}

