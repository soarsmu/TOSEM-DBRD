/*
 * IDetectorExtension.h
 *
 *  Created on: 2010-7-29
 *      Author: Chengnian Sun.
 */

#ifndef IDETECTOREXTENSION_H_
#define IDETECTOREXTENSION_H_

class AbstractBugReport;
class DuplicateBugReport;
class MasterBugReport;

#include <vector>
using namespace std;

class IDetectorExtension {
public:

  virtual void start_processing(const AbstractBugReport& report) = 0;

  virtual void set_comment(const char* comment) = 0;

  virtual void handle_result(
      const vector<const MasterBugReport*>& candidates) = 0;

  virtual void dispose() = 0;

  IDetectorExtension() {
  }

  virtual ~IDetectorExtension() {
  }
};

#endif /* IDETECTOREXTENSION_H_ */
