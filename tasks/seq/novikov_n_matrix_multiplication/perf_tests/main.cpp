#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/novikov_n_matrix_multiplication/include/matrix_mult_seq.hpp"

namespace novikov_n_matrix_multiplication_seq_test {

std::vector<int> getRandomVector(int size_) {
  std::mt19937 gen(0);
  std::uniform_int_distribution<int> dist(-10, 10);
  std::vector<int> v(size_);
  for (int& x : v) x = dist(gen);
  return v;
}

}  // namespace novikov_n_matrix_multiplication_seq_test

TEST(novikov_n_matrix_multiplication_seq, test_pipeline_run) {
  std::vector<int> dimsA = {200, 200};
  std::vector<int> dimsB = {200, 200};
  std::vector<int> A = novikov_n_matrix_multiplication_seq_test::getRandomVector(200 * 200);
  std::vector<int> B = novikov_n_matrix_multiplication_seq_test::getRandomVector(200 * 200);
  std::vector<int> out(200 * 200);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimsA.data()));
  taskData->inputs_count.emplace_back(dimsA.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimsB.data()));
  taskData->inputs_count.emplace_back(dimsB.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs_count.emplace_back(A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskData->inputs_count.emplace_back(B.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<novikov_n_matrix_multiplication_seq::MatrixMultiplicationSequential>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 3;
  auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return static_cast<double>(dt) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  auto check = novikov_n_matrix_multiplication_seq::multiply(200, 200, 200, A, B);
  ASSERT_EQ(check, out);
}

TEST(novikov_n_matrix_multiplication_seq, test_task_run) {
  std::vector<int> dimsA = {200, 200};
  std::vector<int> dimsB = {200, 200};
  std::vector<int> A = novikov_n_matrix_multiplication_seq_test::getRandomVector(200 * 200);
  std::vector<int> B = novikov_n_matrix_multiplication_seq_test::getRandomVector(200 * 200);
  std::vector<int> out(200 * 200);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimsA.data()));
  taskData->inputs_count.emplace_back(dimsA.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimsB.data()));
  taskData->inputs_count.emplace_back(dimsB.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs_count.emplace_back(A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskData->inputs_count.emplace_back(B.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<novikov_n_matrix_multiplication_seq::MatrixMultiplicationSequential>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 3;
  auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return static_cast<double>(dt) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  auto check = novikov_n_matrix_multiplication_seq::multiply(200, 200, 200, A, B);
  ASSERT_EQ(check, out);
}
