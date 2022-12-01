/*
 * IndexType.h
 *
 *  Created on: Mar 8, 2013
 *      Author: neo
 */

#ifndef _INDEXING_TYPE_H_
#define _INDEXING_TYPE_H_

#include <iostream>
#include <sstream>
#include <string>

#include "../util/MacroUtility.h"

using std::iostream;
using std::string;
using std::stringstream;

class IndexingType {
public:
  enum EnumIndexingType {

    NO_INDEXING = 0,

    SUMMARY_INDEXING,

    DESCRIPTION_INDEXING,

    FULL_INDEXING
  };

  static enum EnumIndexingType parse_indexing_type(const int int_type) {
    enum EnumIndexingType type = static_cast<enum EnumIndexingType>(int_type);
    switch (type) {
    case NO_INDEXING:
    case SUMMARY_INDEXING:
    case DESCRIPTION_INDEXING:
    case FULL_INDEXING:
      return type;
    default: {
      char message[100];
      sprintf(message, "un-handled indexing type %d", int_type);
      ERROR_HERE(message);
      return NO_INDEXING;
    }
    }
  }

  static string get_indexing_type_string(const EnumIndexingType type) {
    switch (type) {
    case NO_INDEXING:
      return "NO_INDEXING";
    case SUMMARY_INDEXING:
      return "SUMMARY_INDEXING";
    case DESCRIPTION_INDEXING:
      return "DESCRIPTION_INDEXING";
    case FULL_INDEXING:
      return "FULL_INDEXING";
    default: {
      char message[100];
      sprintf(message, "un-handled indexing type %d", type);
      ERROR_HERE(message);
      return "error";
    }
    }
  }

  static string get_indexing_type_mapping() {
    stringstream ss;
    ss << NO_INDEXING << ":NO_INDEXING, ";
    ss << SUMMARY_INDEXING << ":SUMMARY_INDEXING, ";
    ss << DESCRIPTION_INDEXING << ":DESCRIPTION_INDEXING, ";
    ss << FULL_INDEXING << ":FULL_INDEXING";
    return ss.str();
  }

};

#endif /* _INDEXING_TYPE_H_ */
