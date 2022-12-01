/*
 * PlainSimilarityMeasureFactory.h
 *
 *  Created on: Dec 11, 2010
 *      Author: Chengnian SUN
 */

#ifndef PLAINSIMILARITYMEASUREFACTORY_H_
#define PLAINSIMILARITYMEASUREFACTORY_H_
#include "IPlainSimilarityMeasure.h"
#include "ICSE07_Similarity_Measure.h"
#include "DSN08_Similarity_Measure.h"
#include "ICSE08_Similarity_Measure.h"

#include "combo/ComboICSE07_Similairty_Measure.h"
#include "combo/ComboDSN08_Similarity_Measure.h"
#include "combo/ComboICSE08_Similarity_Measure.h"

#include <string>
#include <iostream>
#include <sstream>
#include <cstdlib>
using namespace std;

class PlainSimilarityMeasureFactory {
public:

  enum SimilarityMeasureType {

    ICSE_07_W1_NO_BIGRAM = 1,

    ICSE_07_W2_NO_BIGRAM = 3,

    DSN_08_W1_NO_BIGRAM = 5,

    DSN_08_W2_NO_BIGRAM = 8,

    ICSE_08_W1_NO_BIGRAM = 9,

    ICSE_08_W2_NO_BIGRAM = 10,

    COMBO_ICSE_07_W1_NO_BIGRAM = 11,

    COMBO_ICSE_07_W2_NO_BIGRAM = 13,

    COMBO_DSN_08_W1_NO_BIGRAM = 15,

    COMBO_DSN_08_W2_NO_BIGRAM = 18,

    COMBO_ICSE_08_W1_NO_BIGRAM = 19,

    COMBO_ICSE_08_W2_NO_BIGRAM = 20,

    NONE = 9999
  };

  static enum SimilarityMeasureType parse_similarity_measure_type(
      const int int_type) {
    enum SimilarityMeasureType type =
        static_cast<SimilarityMeasureType>(int_type);
    switch (type) {
    case ICSE_07_W1_NO_BIGRAM:
      return ICSE_07_W1_NO_BIGRAM;

    case ICSE_07_W2_NO_BIGRAM:
      return ICSE_07_W2_NO_BIGRAM;

    case DSN_08_W1_NO_BIGRAM:
      return DSN_08_W1_NO_BIGRAM;

    case DSN_08_W2_NO_BIGRAM:
      return DSN_08_W2_NO_BIGRAM;

    case ICSE_08_W1_NO_BIGRAM:
      return ICSE_08_W1_NO_BIGRAM;

    case ICSE_08_W2_NO_BIGRAM:
      return ICSE_08_W2_NO_BIGRAM;

    case COMBO_ICSE_07_W1_NO_BIGRAM:
      return COMBO_ICSE_07_W1_NO_BIGRAM;

    case COMBO_ICSE_07_W2_NO_BIGRAM:
      return COMBO_ICSE_07_W2_NO_BIGRAM;

    case COMBO_DSN_08_W1_NO_BIGRAM:
      return COMBO_DSN_08_W1_NO_BIGRAM;

    case COMBO_DSN_08_W2_NO_BIGRAM:
      return COMBO_DSN_08_W2_NO_BIGRAM;

    case COMBO_ICSE_08_W1_NO_BIGRAM:
      return COMBO_ICSE_08_W1_NO_BIGRAM;

    case COMBO_ICSE_08_W2_NO_BIGRAM:
      return COMBO_ICSE_08_W2_NO_BIGRAM;

    default:
      cerr << "ERROR: unhandled SimilarityMeasureType type: " << type << endl;
      exit(1);
      return NONE;
    }
  }

  static string get_similarity_measure_type_string(
      const enum SimilarityMeasureType type) {
    switch (type) {
    case ICSE_07_W1_NO_BIGRAM:
      return "ICSE_07_W1_NO_BIGRAM";
      //		case ICSE_07_W1_WITH_BIGRAM:
      //			return "ICSE_07_W1_WITH_BIGRAM";
      //		case ICSE_07_W2_WITH_BIGRAM:
      //			return "ICSE_07_W2_WITH_BIGRAM";
    case ICSE_07_W2_NO_BIGRAM:
      return "ICSE_07_W2_NO_BIGRAM";

    case DSN_08_W1_NO_BIGRAM:
      return "DSN_08_W1_NO_BIGRAM";

    case DSN_08_W2_NO_BIGRAM:
      return "DSN_08_W2_NO_BIGRAM";

    case ICSE_08_W1_NO_BIGRAM:
      return "ICSE_08_W1_NO_BIGRAM";

    case ICSE_08_W2_NO_BIGRAM:
      return "ICSE_08_W2_NO_BIGRAM";

    case COMBO_ICSE_07_W1_NO_BIGRAM:
      return "COMBO_ICSE_07_W1_NO_BIGRAM";

    case COMBO_ICSE_07_W2_NO_BIGRAM:
      return "COMBO_ICSE_07_W2_NO_BIGRAM";

    case COMBO_DSN_08_W1_NO_BIGRAM:
      return "COMBO_DSN_08_W1_NO_BIGRAM";

    case COMBO_DSN_08_W2_NO_BIGRAM:
      return "COMBO_DSN_08_W2_NO_BIGRAM";

    case COMBO_ICSE_08_W1_NO_BIGRAM:
      return "COMBO_ICSE_08_W1_NO_BIGRAM";

    case COMBO_ICSE_08_W2_NO_BIGRAM:
      return "COMBO_ICSE_08_W2_NO_BIGRAM";

    default:
      cerr << "ERROR: unhandled SimilarityMeasureType type: " << type << endl;
      exit(1);
      return "";
    }
  }

  static string get_similarity_measure_type_mapping() {
    stringstream ss;
    ss << ICSE_07_W1_NO_BIGRAM << ":ICSE_07_W1_NO_BIGRAM, ";
    ss << ICSE_07_W2_NO_BIGRAM << ":ICSE_07_W2_NO_BIGRAM, ";
    ss << DSN_08_W1_NO_BIGRAM << ":DSN_08_W1_NO_BIGRAM, ";
    ss << DSN_08_W2_NO_BIGRAM << ":DSN_08_W2_NO_BIGRAM, ";
    ss << ICSE_08_W1_NO_BIGRAM << ": ICSE_08_W1_NO_BIGRAM, ";
    ss << ICSE_08_W2_NO_BIGRAM << ": ICSE_08_W2_NO_BIGRAM, ";

    ss << COMBO_ICSE_07_W1_NO_BIGRAM << ":COMBO_ICSE_07_W1_NO_BIGRAM, ";
    ss << COMBO_ICSE_07_W2_NO_BIGRAM << ":COMBO_ICSE_07_W2_NO_BIGRAM, ";
    ss << COMBO_DSN_08_W1_NO_BIGRAM << ":COMBO_DSN_08_W1_NO_BIGRAM, ";
    ss << COMBO_DSN_08_W2_NO_BIGRAM << ":COMBO_DSN_08_W2_NO_BIGRAM, ";
    ss << COMBO_ICSE_08_W1_NO_BIGRAM << ": COMBO_ICSE_08_W1_NO_BIGRAM, ";
    ss << COMBO_ICSE_08_W2_NO_BIGRAM << ": COMBO_ICSE_08_W2_NO_BIGRAM, ";
    return ss.str();
  }

  static IPlainSimilarityMeasure* create_similarity_measure(
      enum SimilarityMeasureType type) {
    switch (type) {
    case ICSE_07_W1_NO_BIGRAM:
      return new ICSE07_Similarity_Measure(1);
    case ICSE_07_W2_NO_BIGRAM:
      return new ICSE07_Similarity_Measure(2);
    case DSN_08_W1_NO_BIGRAM:
      return new DSN08_Similarity_Measure(1);
    case DSN_08_W2_NO_BIGRAM:
      return new DSN08_Similarity_Measure(2);
    case ICSE_08_W1_NO_BIGRAM:
      return new ICSE08_Similarity_Measure(1);
    case ICSE_08_W2_NO_BIGRAM:
      return new ICSE08_Similarity_Measure(2);

    case COMBO_ICSE_07_W1_NO_BIGRAM:
      return new ComboICSE07_Similairty_Measure(1);
    case COMBO_ICSE_07_W2_NO_BIGRAM:
      return new ComboICSE07_Similairty_Measure(2);
    case COMBO_DSN_08_W1_NO_BIGRAM:
      return new ComboDSN08_Similarity_Measure(1);
    case COMBO_DSN_08_W2_NO_BIGRAM:
      return new ComboDSN08_Similarity_Measure(2);
    case COMBO_ICSE_08_W1_NO_BIGRAM:
      return new ComboICSE08_Similarity_Measure(1);
    case COMBO_ICSE_08_W2_NO_BIGRAM:
      return new ComboICSE08_Similarity_Measure(2);
    default:
      cerr << "ERROR: cannot reach here." << endl;
      exit(1);
      return NULL;
    }
  }

private:

  PlainSimilarityMeasureFactory() {
  }

  virtual ~PlainSimilarityMeasureFactory() {
  }

};

#endif /* PLAINSIMILARITYMEASUREFACTORY_H_ */
