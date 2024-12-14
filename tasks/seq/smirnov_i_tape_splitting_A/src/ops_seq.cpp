#include "seq/smirnov_i_tape_splitting_A/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool smirnov_i_tape_splitting_A::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  A = new double[m_a * n_a];
  B = new double[m_b * n_b];
  auto* tmp_ptr_a = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* tmp_ptr_b = reinterpret_cast<double*>(taskData->inputs[1]);
  for (int i = 0; i < m_a * n_a; i++) {
    A[i] = tmp_ptr_a[i];
  }
  for (int i = 0; i < m_b * n_b; i++) {
    B[i] = tmp_ptr_b[i];
  }

  // Init value for output

  res = nullptr;
  return true;
}

bool smirnov_i_tape_splitting_A::TestMPITaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count.size() != 4) return false;
  m_a = taskData->inputs_count[0];
  n_a = taskData->inputs_count[1];
  m_b = taskData->inputs_count[2];
  n_b = taskData->inputs_count[3];
  return (n_a == m_b && m_a > 0 && n_a > 0 && m_b > 0 && n_b > 0);
}

bool smirnov_i_tape_splitting_A::TestMPITaskSequential::run() {
  internal_order_test();
  res = new double[m_a * n_b];
  std::fill(res, res + m_a * n_b, 0.0);
  for (int i = 0; i < m_a; i++) {
    for (int j = 0; j < n_b; j++) {
      for (int k = 0; k < n_a; k++) {
        res[i * n_b + j] += A[i * n_a + k] * B[k * n_b + j];
      }
    }
  }

  return true;
}

bool smirnov_i_tape_splitting_A::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(res, res + m_a * n_b, tmp_ptr);
  return true;
}
