/*
 * SectionType.h
 *
 *  Created on: Feb 1, 2011
 *      Author: neo
 */

#ifndef SECTION_TYPE_H__
#define SECTION_TYPE_H__

class SectionType {
public:
  enum EnumSectionType {
    SUM_UNI = 0, SUM_BI = 1, SUM_TRI = 2,

    DESC_UNI = 3, DESC_BI = 4, DESC_TRI = 5,

    ALL_UNI = 6, ALL_BI = 7, ALL_TRI = 8
  };
};

const char* get_section_type_name(const enum SectionType::EnumSectionType type);

#endif /* SECTION_TYPE_H__ */
