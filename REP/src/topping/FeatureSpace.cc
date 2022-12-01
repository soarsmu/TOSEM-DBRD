#include <boost/unordered_map.hpp>
#include <cassert>

#include "../libsvm/svm.h"
#include "../util/MacroUtility.h"

#include "FeatureSpace.h"

using namespace std;

#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

#define MAX_DOUBLE_VALUE 999999999

#define DEFAULT_LOWER_BOUND -1
#define DEFAULT_UPPER_BOUND 1

//----------------------------------------------------------------------------------

struct FeatureVectorWrapper {
  svm_node* feature_vector;
  int index_in_feature_space;
  int label;

  FeatureVectorWrapper(svm_node* feature_vector, int index, int label) :
      feature_vector(feature_vector), index_in_feature_space(index), label(
          label) {
  }

  FeatureVectorWrapper() :
      feature_vector(NULL), index_in_feature_space(0), label(0) {
  }
};

class FeatureVectorHasher {
private:
  inline size_t hash_code(svm_node* a) const {
    //if (length == 0) {
    //	return 0;
    //}
    size_t result = 1;
    assert(2 * sizeof(unsigned int) == sizeof(double));
    for (int i = 0; a[i].index != -1; i++) {
      //			double value = a[i].value;
      unsigned int* bits = reinterpret_cast<unsigned int*>(&(a[i].value));
      result = 31 * result + static_cast<size_t>(bits[1] ^ (bits[0]));
    }
    return result;
  }

public:
  size_t operator()(const FeatureVectorWrapper& t) const {
    size_t prime = 31;
    size_t result = 1;
    result = prime * result + hash_code(t.feature_vector);
    return result;
  }
};

struct EqualFeatureVector {
  bool operator()(const FeatureVectorWrapper& wrapper_1,
      const FeatureVectorWrapper& wrapper_2) const {
    int i;
    svm_node* feature_vector_1 = wrapper_1.feature_vector;
    svm_node* feature_vector_2 = wrapper_2.feature_vector;
    for (i = 0;
        feature_vector_1[i].index != -1 && feature_vector_2[i].index != -1;
        i++) {
      if (feature_vector_1[i].value != feature_vector_2[i].value) {
        return false;
      }
    }
    if (feature_vector_1[i].index == -1 && feature_vector_2[i].index == -1) {
      return true;
    } else {
      assert(false);
      return false;
    }

  }
};

//----------------------------------------------------------------------------------

void FeatureSpace::prune_duplicate_feature_vectors() {
  //boost::unordered_map<FeatureVectorWrapper, FeatureVectorWrapper, FeatureVectorHasher, EqualFeatureVector> set;
  boost::unordered_map<FeatureVectorWrapper, FeatureVectorWrapper,
      FeatureVectorHasher, EqualFeatureVector> map;

  //	const unsigned int number_of_feature_vectors = this->feature_space.size();

  //int number_of_relevant = 0;
  //int number_of_irrelevant = 0;

  for (int i = this->feature_space.size() - 1; i > -1; i--) {
    //if (this->labels[i] == 0) {
    //	number_of_irrelevant++;
    //} else {
    //	number_of_relevant++;
    //}
    // all feature vectors should be valid.
    assert(this->validity_indicators[i]);
    this->validity_indicators[i] = false;

    //svm_node* feature_vector = this->feature_space[i];
    FeatureVectorWrapper wrapper(this->feature_space[i], i, this->labels[i]);
    boost::unordered_map<FeatureVectorWrapper, FeatureVectorWrapper,
        FeatureVectorHasher, EqualFeatureVector>::iterator iter = map.find(
        wrapper);
    //boost::unordered_set<FeatureVectorWrapper, FeatureVectorHasher, EqualFeatureVector>::iterator iter = set.find(wrapper);
    if (iter == map.end()) {
      map[wrapper] = wrapper;
    } else {
      if (iter->second.label != wrapper.label) {
        if (iter->second.label == 0) {
          map[iter->first] = wrapper;
        }
      }
    }
  }

  //TRACE(cout << "relevant   = " << number_of_relevant << '\n');
  //TRACE(cout << "irrelevant = " << number_of_irrelevant << '\n');

  for (boost::unordered_map<FeatureVectorWrapper, FeatureVectorWrapper,
      FeatureVectorHasher, EqualFeatureVector>::iterator iter = map.begin();
      iter != map.end(); iter++) {
    this->validity_indicators[iter->first.index_in_feature_space] = true;
  }
}

void FeatureSpace::learn_scaling_parameters() {
  this->min_per_feature.assign(this->feature_count, MAX_DOUBLE_VALUE);
  this->max_per_feature.assign(this->feature_count, -MAX_DOUBLE_VALUE);

  const unsigned int number_of_vectors = this->feature_space.size();
  for (unsigned int i = 0; i < number_of_vectors; i++) {
    if (this->validity_indicators[i] == false) {
      continue;
    }
    svm_node* feature_vector = this->feature_space[i];
    for (int j = 0; j < this->feature_count; j++) {
      min_per_feature[j] = MIN(min_per_feature[j], feature_vector[j].value);
      max_per_feature[j] = MAX(max_per_feature[j], feature_vector[j].value);
    }
  }
}

/*
 if(feature_max[index] == feature_min[index])
 return;

 if(value == feature_min[index])
 value = lower;
 else if(value == feature_max[index])
 value = upper;
 else
 value = lower + (upper-lower) *
 (value-feature_min[index])/
 (feature_max[index]-feature_min[index]);

 if(value != 0)
 {
 printf("%d:%g ",index, value);
 new_num_nonzeros++;
 }
 */

void FeatureSpace::scale_feature_space() {
  const unsigned int number_of_vectors = this->feature_space.size();
  for (unsigned int i = 0; i < number_of_vectors; i++) {
    if (this->validity_indicators[i] == false) {
      continue;
    }
    svm_node* feature_vector = this->feature_space[i];
    for (int j = 0; j < this->feature_count; j++) {
      svm_node& node = feature_vector[j];
      double min = min_per_feature[j];
      double max = max_per_feature[j];
      if (min == max) {
        continue;
      }
      if (node.value == min) {
        node.value = DEFAULT_LOWER_BOUND;
      } else if (node.value == max) {
        node.value = DEFAULT_UPPER_BOUND;
      } else {
        node.value = DEFAULT_LOWER_BOUND
            + (DEFAULT_UPPER_BOUND - DEFAULT_LOWER_BOUND) * (node.value - min)
                / (max - min);
      }
    }
  }
}

void FeatureSpace::scale_testing_vector(svm_node* testing_feature_vector) {
  //assert(this->feature_count == number_of_features);
  for (int j = 0; j < this->feature_count; j++) {
    svm_node& node = testing_feature_vector[j];
    double min = min_per_feature[j];
    double max = max_per_feature[j];
    if (min == max) {
      continue;
    }
    if (node.value == min) {
      node.value = DEFAULT_LOWER_BOUND;
    } else if (node.value == max) {
      node.value = DEFAULT_UPPER_BOUND;
    } else {
      node.value = DEFAULT_LOWER_BOUND
          + (DEFAULT_UPPER_BOUND - DEFAULT_LOWER_BOUND) * (node.value - min)
              / (max - min);
    }
  }
}

svm_node * FeatureSpace::get_valid_feature_vector(const unsigned int index) {
  assert(index < this->validity_indicators.size());
  assert(this->validity_indicators[index]);
  return this->feature_space[index];
}

svm_node * FeatureSpace::get_feature_vector(const unsigned int index) {
  //	svm_node node;

  for (unsigned int i = this->feature_space.size(); i <= index; i++) {
    svm_node* feature_vector =
        new svm_node[this->get_length_of_feature_vector()];
    this->feature_space.push_back(feature_vector);
    this->labels.push_back(0);
    this->validity_indicators.push_back(false);

    for (int j = 0; j < this->feature_count; j++) {
      /*node.index = j;
       node.value = 0;*/
      //this->feature_space[i].push_back(0);
      feature_vector[j].index = j;
    }
    feature_vector[this->feature_count].index = -1;
    /*	node.index = -1;
     this->feature_space[i].push_back(node);*/
  }
  assert(this->validity_indicators[index] == false);
  this->validity_indicators[index] = true;
  return (this->feature_space[index]);
}

void FeatureSpace::init(const unsigned int feature_count) {
  this->feature_count = feature_count;
}

FeatureSpace::FeatureSpace() {
  this->feature_count = -1;
}

FeatureSpace::~FeatureSpace(void) {
  const unsigned int number_of_feature_vectors = this->feature_space.size();
  for (unsigned int i = 0; i < number_of_feature_vectors; i++) {
    delete[] this->feature_space[i];
    this->feature_space[i] = NULL;
  }
}
