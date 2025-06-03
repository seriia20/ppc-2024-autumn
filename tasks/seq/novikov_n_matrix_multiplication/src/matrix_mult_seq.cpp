#include "seq/novikov_n_matrix_multiplication/include/matrix_mult_seq.hpp"

#include <thread>

std::vector<int> novikov_n_matrix_multiplication_seq::multiply(size_t M, size_t N, size_t K, const std::vector<int>& A,
                                                               const std::vector<int>& B) {
  std::vector<int> C(M * N, 0);
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      for (size_t k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
  return C;
}

bool novikov_n_matrix_multiplication_seq::MatrixMultiplicationSequential::pre_processing() {
  internal_order_test();

  auto* dimsA = reinterpret_cast<int*>(taskData->inputs[0]);
  M_ = static_cast<size_t>(dimsA[0]);
  K_ = static_cast<size_t>(dimsA[1]);

  auto* dimsB = reinterpret_cast<int*>(taskData->inputs[1]);
  size_t lineB = static_cast<size_t>(dimsB[0]);
  N_ = static_cast<size_t>(dimsB[1]);

  A_.assign(reinterpret_cast<int*>(taskData->inputs[2]),
            reinterpret_cast<int*>(taskData->inputs[2]) + taskData->inputs_count[2]);
  B_.assign(reinterpret_cast<int*>(taskData->inputs[3]),
            reinterpret_cast<int*>(taskData->inputs[3]) + taskData->inputs_count[3]);

  C_.resize(M_ * N_);

  return lineB == K_;
}

bool novikov_n_matrix_multiplication_seq::MatrixMultiplicationSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count.size() != 4 || taskData->outputs_count.size() != 1) {
    return false;
  }
  if (taskData->inputs_count[0] != 2 || taskData->inputs_count[1] != 2) {
    return false;
  }

  auto* dimsA = reinterpret_cast<int*>(taskData->inputs[0]);
  auto* dimsB = reinterpret_cast<int*>(taskData->inputs[1]);

  size_t m = static_cast<size_t>(dimsA[0]);
  size_t k = static_cast<size_t>(dimsA[1]);
  size_t lb = static_cast<size_t>(dimsB[0]);
  size_t n = static_cast<size_t>(dimsB[1]);

  if (m == 0 || k == 0 || lb == 0 || n == 0) {
    return false;
  }

  if (k != lb) {
    return false;
  }

  if (taskData->inputs_count[2] != m * k) {
    return false;
  }

  if (taskData->inputs_count[3] != lb * n) {
    return false;
  }

  if (taskData->outputs_count[0] != m * n) {
    return false;
  }

  return true;
}

bool novikov_n_matrix_multiplication_seq::MatrixMultiplicationSequential::run() {
  internal_order_test();
  C_ = multiply(M_, N_, K_, A_, B_);
  return true;
}

bool novikov_n_matrix_multiplication_seq::MatrixMultiplicationSequential::post_processing() {
  internal_order_test();
  auto* out = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < M_ * N_; ++i) {
    out[i] = C_[i];
  }
  return true;
}
