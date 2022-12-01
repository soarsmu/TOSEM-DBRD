/*
 * RandomUtility.h
 *
 *  Created on: Dec 30, 2010
 *      Author: Chengnian SUN
 */

#ifndef RANDOMUTILITY_H_
#define RANDOMUTILITY_H_
#include <cstdlib>
#include <vector>
using namespace std;

class RandomUtility {
private:
  static RandomUtility* INSTANCE;

public:

  static RandomUtility& get_default();

  static void dispose_default();

  unsigned re_seed();

  /**
   * [0, 1]
   */
  double random_double() {
    return static_cast<double>(rand()) / RAND_MAX;
  }

private:
  RandomUtility();

  ~RandomUtility() {
  }

  vector<unsigned> m_seeds;

  size_t m_index;

  unsigned next_seed();

};

#endif /* RANDOMUTILITY_H_ */
