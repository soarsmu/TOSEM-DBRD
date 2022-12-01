/*
 * Version.cc
 *
 *  Created on: Mar 14, 2013
 *      Author: neo
 */

#include <string>

#include "Version.h"

using std::string;

string get_version_string() {
//  string s = "Major ";
  string s = "Version ";
  s += MAJOR_VERSION;
//  s += ", Minor ";
  s += ".";
  s += MINOR_VERSION;
  s += ", Built on ";
  s += BUILDING_TIME;
  return s;
}
