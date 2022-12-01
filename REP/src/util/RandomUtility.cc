/*
 * RandomUtility.cc
 *
 *  Created on: Dec 30, 2010
 *      Author: Chengnian SUN
 */

#include "RandomUtility.h"

#include <cstdio>
#include <cstdlib>
#include <time.h>

RandomUtility* RandomUtility::INSTANCE = NULL;

void RandomUtility::dispose_default() {
  delete RandomUtility::INSTANCE;
  RandomUtility::INSTANCE = NULL;
}

RandomUtility& RandomUtility::get_default() {
  if (!RandomUtility::INSTANCE) {
    RandomUtility::INSTANCE = new RandomUtility();
  }
  return *RandomUtility::INSTANCE;
}

unsigned RandomUtility::re_seed() {
  const unsigned seed = next_seed();
  srand(seed);
  return seed;
}

unsigned RandomUtility::next_seed() {
  if (this->m_index < this->m_seeds.size()) {
    return this->m_seeds[this->m_index++];
  } else {
    fprintf(stderr, "You can only get at most %u distinct random seeds.\n",
        static_cast<unsigned>(this->m_seeds.size()));
    exit(1);
    return 0;
  }
}

RandomUtility::RandomUtility() {
  m_index = 0;
  srand(time(NULL));
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(rand());
  m_seeds.push_back(104878398);
  m_seeds.push_back(19885835);
  m_seeds.push_back(139618234);
  m_seeds.push_back(59662956);
  m_seeds.push_back(203396585);
  m_seeds.push_back(142818943);
  m_seeds.push_back(102074960);
  m_seeds.push_back(62762486);
  m_seeds.push_back(195982352);
  m_seeds.push_back(213453361);
  m_seeds.push_back(72404736);
  m_seeds.push_back(77283859);
  m_seeds.push_back(109301425);
  m_seeds.push_back(76457700);
  m_seeds.push_back(3057844);
  m_seeds.push_back(89064550);
  m_seeds.push_back(205854101);
  m_seeds.push_back(156925996);
  m_seeds.push_back(139619380);
  m_seeds.push_back(115834838);
  m_seeds.push_back(83557963);
  m_seeds.push_back(43658190);
  m_seeds.push_back(110398722);
  m_seeds.push_back(106250898);
  m_seeds.push_back(63749540);
  m_seeds.push_back(34711086);
  m_seeds.push_back(201117633);
  m_seeds.push_back(19113722);
  m_seeds.push_back(106491390);
  m_seeds.push_back(110458522);
  m_seeds.push_back(130442070);
  m_seeds.push_back(150181447);
  m_seeds.push_back(115929174);
  m_seeds.push_back(96410206);
  m_seeds.push_back(86955642);
  m_seeds.push_back(211638816);
  m_seeds.push_back(8074412);
  m_seeds.push_back(163303168);
  m_seeds.push_back(31241449);
  m_seeds.push_back(118351240);
  m_seeds.push_back(186703299);
  m_seeds.push_back(21112608);
  m_seeds.push_back(40711421);
  m_seeds.push_back(160570759);
  m_seeds.push_back(98069886);
  m_seeds.push_back(20732974);
  m_seeds.push_back(121123858);
  m_seeds.push_back(185151544);
  m_seeds.push_back(82835875);
  m_seeds.push_back(36649857);
  m_seeds.push_back(67342637);
  m_seeds.push_back(25556526);
  m_seeds.push_back(172454030);
  m_seeds.push_back(126974632);
  m_seeds.push_back(20525153);
  m_seeds.push_back(207531151);
  m_seeds.push_back(19312131);
  m_seeds.push_back(105196372);
  m_seeds.push_back(168870917);
  m_seeds.push_back(74898827);
  m_seeds.push_back(197784369);
  m_seeds.push_back(180050453);
  m_seeds.push_back(13834489);
  m_seeds.push_back(131138933);
  m_seeds.push_back(102561378);
  m_seeds.push_back(149252778);
  m_seeds.push_back(180044572);
  m_seeds.push_back(105672876);
  m_seeds.push_back(90148973);

}
