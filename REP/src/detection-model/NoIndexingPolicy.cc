/*
 * NoIndexingPolicy.cc
 *
 *  Created on: Mar 9, 2013
 *      Author: neo
 */

#include <cstdlib>

#include "NoIndexingPolicy.h"

NoIndexingPolicy::NoIndexingPolicy() :
    AbstractIndexingPolicy(NULL) {

}

NoIndexingPolicy::~NoIndexingPolicy() {
}

void NoIndexingPolicy::update_index(AbstractBugReport&) const {

}

