/*
 * TimestampMap.cc
 *
 *  Created on: Apr 8, 2013
 *      Author: neo
 */

#include <cstdio>
#include "../util/MacroUtility.h"
#include "TimestampMap.h"

TimestampMap::TimestampMap(string path) :
    m_map(path.empty() ? NULL : new MAP()) {

  if (path.empty()) {
    return;
  }


  FILE* file = fopen(path.c_str(), "r");
  if (!file) {
    string msg = "Cannot open time stamp file ";
    msg += path;
    EXIT_ON_ERROR(msg.c_str());
  }

  unsigned id;
  unsigned timestamp;

  int result;
  while ((result = fscanf(file, "%u=%u", &id, &timestamp)) >= 0) {
    if (!result) {
      continue;
    }
    if (result != 2) {
      printf("cannot reach here..\n");
    }
    (*this->m_map)[id] = timestamp;
  }
  fclose(file);
}

TimestampMap::~TimestampMap() {
  delete this->m_map;
}

unsigned TimestampMap::get_timestamp(const unsigned report_id) const {
  if (this->m_map) {
    MAP::const_iterator iter = this->m_map->find(report_id);
    if (iter != this->m_map->end()) {
      return iter->second;
    } else {
      char msg[400];
      sprintf(msg, "Cannot find timestamp for report %u", report_id);
      EXIT_ON_ERROR(msg);
      return 0;
    }
  } else {
    return 0;
  }

}

