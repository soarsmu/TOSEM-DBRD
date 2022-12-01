/*
 * RankNetTrainingInstance.h
 *
 *  Created on: Jan 10, 2011
 *      Author: Chengnian SUN
 */

#ifndef RANKNETTRAININGINSTANCE_H_
#define RANKNETTRAININGINSTANCE_H_

class AbstractBugReport;

class RankNetTrainingInstance {
private:
  const AbstractBugReport* m_query;
  const AbstractBugReport* m_relevant_report;
  const AbstractBugReport* m_irrelevant_report;

public:

  inline RankNetTrainingInstance(const AbstractBugReport* query_report,
      const AbstractBugReport* relevant_report,
      const AbstractBugReport* irrelevant_report) :
      m_query(query_report), m_relevant_report(relevant_report), m_irrelevant_report(
          irrelevant_report) {
  }

  inline const AbstractBugReport& get_query() const {
    return *(this->m_query);
  }

  inline const AbstractBugReport& get_relevant_report() const {
    return *(this->m_relevant_report);
  }

  inline const AbstractBugReport& get_irrelevant_report() const {
    return *(this->m_irrelevant_report);
  }

  ~RankNetTrainingInstance() {

  }
};

#endif /* RANKNETTRAININGINSTANCE_H_ */
