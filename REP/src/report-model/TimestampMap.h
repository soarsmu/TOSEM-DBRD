/*
 * TimestampMap.h
 *
 *  Created on: Apr 8, 2013
 *      Author: neo
 */

#ifndef _TIME_STAMP_MAP_H_
#define _TIME_STAMP_MAP_H_

#include <boost/unordered_map.hpp>
#include <cassert>
#include <string>

using boost::unordered_map;
using std::string;

class TimestampMap {
private:
  typedef boost::unordered_map<unsigned, unsigned> MAP;

  MAP* const m_map;

public:

  explicit TimestampMap(string path = "");

  unsigned get_timestamp(unsigned report_id) const;

  ~TimestampMap();

};

#endif /* _TIME_STAMP_MAP_H_ */
