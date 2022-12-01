#include "BM25FParameter.h"

void BM25FParameter::initialize(const float total_weight,
    const bool total_weight_fixed, const float k1, const bool k1_fixed,
    const float summary_weight, const bool summary_weight_fixed, const
    float summary_b, const bool summary_b_fixed,
    const float description_weight, const bool description_weight_fixed,
    const float description_b, const bool description_b_fixed, const float k3,
    const bool k3_fixed) {
  this->m_total_weight.value = total_weight;
  this->m_total_weight.fixed = total_weight_fixed;

  this->m_k1.value = k1;
  this->m_k1.fixed = k1_fixed;

  this->m_summary_weight.value = summary_weight;
  this->m_summary_weight.fixed = summary_weight_fixed;

  this->m_summary_b.value = summary_b;
  this->m_summary_b.fixed = summary_b_fixed;

  this->m_description_weight.value = description_weight;
  this->m_description_weight.fixed = description_weight_fixed;

  this->m_description_b.value = description_b;
  this->m_description_b.fixed = description_b_fixed;

  this->m_k3.value = k3;
  this->m_k3.fixed = k3_fixed;
}

void BM25FParameter::print(FILE* file, const char* title) const {
  fprintf(file, "\n[%s] BM25F Parameter...\n", title);
  fprintf(file, "[total weight, fixed = %d] = %f\n", this->m_total_weight.fixed,
      this->m_total_weight.value);
  fprintf(file, "[k1, fixed = %d] = %f\n", this->m_k1.fixed, this->m_k1.value);
  fprintf(file, "[summary weight, fixed = %d] = %f\n",
      this->m_summary_weight.fixed, this->m_summary_weight.value);
  fprintf(file, "[summary b, fixed = %d] = %f\n", this->m_summary_b.fixed,
      this->m_summary_b.value);
  fprintf(file, "[description weight, fixed = %d] = %f\n",
      this->m_description_weight.fixed, this->m_description_weight.value);
  fprintf(file, "[description b, fixed = %d] = %f\n",
      this->m_description_b.fixed, this->m_description_b.value);
  fprintf(file, "[k3, fixed = %d] = %f\n", this->m_k3.fixed, this->m_k3.value);
}
