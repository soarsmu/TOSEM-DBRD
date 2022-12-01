#ifndef FEATURE_SPACE_
#define FEATURE_SPACE_

#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

using namespace std;

struct svm_node;

// this class is the feature space for the training set.
class FeatureSpace {
private:
  vector<svm_node*> feature_space;

  vector<int> labels;

  vector<bool> validity_indicators;

  int feature_count;

  // store min value of each feature. for scaling
  vector<double> min_per_feature;

  // store max value of each feature. for scaling
  vector<double> max_per_feature;

public:

  inline void reset_validity_indicators() {
    const unsigned int size = this->validity_indicators.size();
    for (unsigned int i = 0; i < size; i++) {
      this->validity_indicators[i] = false;
    }
  }

  //inline void disable_feature_vector(const unsigned int index) {
  //	this->validity_indicators[index] = false;
  //}

  void prune_duplicate_feature_vectors();

  void learn_scaling_parameters();

  void scale_feature_space();

  void scale_testing_vector(svm_node* testing_feature_vector);

  inline unsigned get_length_of_feature_vector() const {
    return this->feature_count + 1;
  }

  inline void set_label(const unsigned int index, int label) {
    this->labels[index] = label;
  }

  inline unsigned int get_number_of_valid_feature_vectors() {
    unsigned int result = 0;
    const unsigned int size = this->validity_indicators.size();
    for (unsigned int i = 0; i < size; i++) {
      if (this->validity_indicators[i]) {
        result++;
      }
    }
    return result;
  }

  inline pair<int, double*> new_and_init_label_array() {
    const unsigned int size = this->get_number_of_valid_feature_vectors();
    double* labels = new double[size];

    unsigned int vector_index = 0;
    for (unsigned int i = 0; i < size; i++) {
      while (this->validity_indicators[vector_index] == false) {
        vector_index++;
        assert(vector_index < this->labels.size());
      }
      labels[i] = this->labels[vector_index++];
    }
    pair<int, double*> p(size, labels);
    return p;
  }

  svm_node* get_feature_vector(const unsigned int index);

  svm_node* get_valid_feature_vector(const unsigned int index);

  inline bool is_valid_feature_vector(const unsigned int index) {
    return this->validity_indicators[index];
  }

  void init(const unsigned int feature_count);

  FeatureSpace();

  ~FeatureSpace(void);
};

#endif /*FEATURE_SPACE_*/
