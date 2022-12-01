/*
 * SectionAndTerm.cc
 *
 *  Created on: Feb 1, 2011
 *      Author: neo
 */

#include <cassert>
#include <cstdlib>
#include <iostream>
using namespace std;

#include "SectionType.h"

const char* get_section_type_name(
    const enum SectionType::EnumSectionType type) {
  switch (type) {
  case SectionType::SUM_UNI:
    return "SUM_UNI";
  case SectionType::SUM_BI:
    return "SUM_BI";
  case SectionType::SUM_TRI:
    return "SUM_TRI";

  case SectionType::DESC_UNI:
    return "DESC_UNI";
  case SectionType::DESC_BI:
    return "DESC_BI";
  case SectionType::DESC_TRI:
    return "DESC_TRI";

  case SectionType::ALL_UNI:
    return "ALL_UNI";
  case SectionType::ALL_BI:
    return "ALL_BI";
  case SectionType::ALL_TRI:
    return "ALL_TRI";
  default:
    assert(false);
    cerr << "ERROR: cannot reach here at line " << __LINE__ << ", " << __FILE__
        << endl;
    exit(1);
    return "<ERROR>";
  }
}
