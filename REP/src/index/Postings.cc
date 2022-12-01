/*
 * Postings.cc
 *
 *  Created on: Mar 8, 2013
 *      Author: neo
 */

#include "Postings.h"

#include <cassert>
#include <cstdio>

#include "../report-model/AbstractBugReport.h"
#include "../util/MacroUtility.h"

Postings::Postings(const Postings& copy) :
    m_term_id(copy.m_term_id), m_reports(copy.m_reports) {
  UNREACHABLE("not implemented");
}

Postings& Postings::operator=(const Postings&) {
  UNREACHABLE("not implemented");
  return *this;
}

Postings::~Postings() {
}

void Postings::add_report(const float normalized_tf,
    const AbstractBugReport& report) {
  assert(
      this->m_reports.empty() || this->m_reports.back().get_report()->get_id() < report.get_id());
  this->m_reports.push_back(TermInReport(&report, normalized_tf));
}
