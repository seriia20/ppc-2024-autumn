// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/borisov_s_crs_mul/include/ops_seq.hpp"

static void dense_to_crs(const std::vector<double>& dense, int M, int N, std::vector<double>& values,
                         std::vector<int>& col_index, std::vector<int>& row_ptr) {
  row_ptr.resize(M + 1, 0);
  std::vector<double> tmp_vals;
  std::vector<int> tmp_cols;

  for (int i = 0; i < M; i++) {
    int count = 0;
    for (int j = 0; j < N; j++) {
      double val = dense[(i * N) + j];
      if (val != 0.0) {
        tmp_vals.push_back(val);
        tmp_cols.push_back(j);
        count++;
      }
    }
    row_ptr[i] = (int)tmp_vals.size() - count;
  }
  row_ptr[M] = (int)tmp_vals.size();

  values = tmp_vals;
  col_index = tmp_cols;
}

static void generate_dense_matrix(int M, int N, double density, std::vector<double>& dense) {
  std::mt19937_64 gen(42);
  std::uniform_real_distribution<double> dist_val(0.1, 10.0);
  std::uniform_real_distribution<double> dist_density(0.0, 1.0);

  dense.resize(M * N, 0.0);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (dist_density(gen) < density) {
        dense[(i * N) + j] = dist_val(gen);
      } else {
        dense[(i * N) + j] = 0.0;
      }
    }
  }
}

static std::shared_ptr<ppc::core::TaskData> create_task_data(
    const std::vector<double>& A_values, const std::vector<int>& A_col, const std::vector<int>& A_row,
    const std::vector<double>& B_values, const std::vector<int>& B_col, const std::vector<int>& B_row,
    size_t C_values_size, size_t C_col_index_size, size_t C_row_ptr_size) {
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs = {reinterpret_cast<uint8_t*>(const_cast<double*>(A_values.data())),
                      reinterpret_cast<uint8_t*>(const_cast<int*>(A_col.data())),
                      reinterpret_cast<uint8_t*>(const_cast<int*>(A_row.data())),
                      reinterpret_cast<uint8_t*>(const_cast<double*>(B_values.data())),
                      reinterpret_cast<uint8_t*>(const_cast<int*>(B_col.data())),
                      reinterpret_cast<uint8_t*>(const_cast<int*>(B_row.data()))};

  taskData->inputs_count = {static_cast<unsigned int>(A_values.size()), static_cast<unsigned int>(A_col.size()),
                            static_cast<unsigned int>(A_row.size()),    static_cast<unsigned int>(B_values.size()),
                            static_cast<unsigned int>(B_col.size()),    static_cast<unsigned int>(B_row.size())};

  std::vector<double> C_val(C_values_size, 0.0);
  std::vector<int> C_col_index(C_col_index_size, 0);
  std::vector<int> C_row_ptr(C_row_ptr_size, 0);

  taskData->outputs = {reinterpret_cast<uint8_t*>(C_val.data()), reinterpret_cast<uint8_t*>(C_col_index.data()),
                       reinterpret_cast<uint8_t*>(C_row_ptr.data())};

  taskData->outputs_count = {static_cast<unsigned int>(C_val.size()), static_cast<unsigned int>(C_col_index.size()),
                             static_cast<unsigned int>(C_row_ptr.size())};

  return taskData;
}

TEST(CrsCrsMulValidation, SuccessfulCase) {
  int M = 2;
  int N = 2;
  int K = 2;
  std::vector<double> A_dense = {1, 2, 3, 4};
  std::vector<double> B_dense = {5, 6, 7, 8};

  std::vector<double> A_values;
  std::vector<int> A_col_index;
  std::vector<int> A_row_ptr;
  dense_to_crs(A_dense, M, N, A_values, A_col_index, A_row_ptr);

  std::vector<double> B_values;
  std::vector<int> B_col_index;
  std::vector<int> B_row_ptr;
  dense_to_crs(B_dense, N, K, B_values, B_col_index, B_row_ptr);

  auto taskData = create_task_data(A_values, A_col_index, A_row_ptr, B_values, B_col_index, B_row_ptr, 4, 4, M + 1);

  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(CrsCrsMulValidation, WrongNumberOfInputs) {
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.resize(5);
  taskData->inputs_count.resize(5);

  taskData->outputs.resize(3);
  taskData->outputs_count.resize(3);

  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(CrsCrsMulValidation, WrongNumberOfOutputs) {
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.resize(6);
  taskData->inputs_count.resize(6);

  taskData->outputs.resize(2);
  taskData->outputs_count.resize(2);

  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(CrsCrsMulValidation, WrongCountsSize) {
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.resize(6);
  taskData->outputs.resize(3);
  taskData->inputs_count.resize(5);
  taskData->outputs_count.resize(2);

  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(CrsCrsMulValidation, IncompatibleDimensions) {
  int M = 2;
  int N = 3;
  int K = 2;
  double density = 0.5;

  std::vector<double> A_dense;
  generate_dense_matrix(M, N, density, A_dense);

  std::vector<double> B_dense;
  generate_dense_matrix(N, K, density, B_dense);

  generate_dense_matrix(2, 2, density, B_dense);

  std::vector<double> A_values;
  std::vector<int> A_col;
  std::vector<int> A_row;
  dense_to_crs(A_dense, M, N, A_values, A_col, A_row);

  std::vector<double> B_values;
  std::vector<int> B_col;
  std::vector<int> B_row;
  dense_to_crs(B_dense, 2, 2, B_values, B_col, B_row);

  auto taskData = create_task_data(A_values, A_col, A_row, B_values, B_col, B_row, 10, 10, M + 1);
  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(CrsCrsMulValidation, InsufficientCRowPtrLength) {
  int M = 2;
  int N = 2;
  int K = 2;
  std::vector<double> A_dense = {1, 2, 3, 4};
  std::vector<double> B_dense = {5, 6, 7, 8};

  std::vector<double> A_values;
  std::vector<int> A_col_index;
  std::vector<int> A_row_ptr;
  dense_to_crs(A_dense, M, N, A_values, A_col_index, A_row_ptr);

  std::vector<double> B_values;
  std::vector<int> B_col_index;
  std::vector<int> B_row_ptr;
  dense_to_crs(B_dense, N, K, B_values, B_col_index, B_row_ptr);

  auto taskData = create_task_data(A_values, A_col_index, A_row_ptr, B_values, B_col_index, B_row_ptr, 4, 4, 2);

  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(CrsCrsMulValidation, NegativeRowCount) {
  std::vector<double> A_dense;
  std::vector<double> A_values;
  std::vector<int> A_col_index;
  std::vector<int> A_row_ptr = {};

  int MB = 1;
  int NB = 1;
  std::vector<double> B_dense = {10};
  std::vector<double> B_values;
  std::vector<int> B_col_index;
  std::vector<int> B_row_ptr;
  dense_to_crs(B_dense, MB, NB, B_values, B_col_index, B_row_ptr);

  auto taskData = create_task_data(A_values, A_col_index, A_row_ptr, B_values, B_col_index, B_row_ptr, 10, 10, 2);
  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(CrsCrsMulValidation, ZeroNcols) {
  int M = 2;
  int N = 2;
  std::vector<double> A_dense(M * N, 0.0);
  std::vector<double> A_values;
  std::vector<int> A_col_index;
  std::vector<int> A_row_ptr;
  dense_to_crs(A_dense, M, N, A_values, A_col_index, A_row_ptr);

  std::vector<double> B_dense(M * N, 0.0);
  std::vector<double> B_values;
  std::vector<int> B_col_index;
  std::vector<int> B_row_ptr;
  dense_to_crs(B_dense, N, N, B_values, B_col_index, B_row_ptr);

  auto taskData = create_task_data(A_values, A_col_index, A_row_ptr, B_values, B_col_index, B_row_ptr, 4, 4, M + 1);
  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(CrsCrsMulValidation, LargeRandom1000x1000) {
  int M = 1000;
  int N = 1000;
  int K = 1000;
  double density = 0.001;

  std::vector<double> A_dense;
  generate_dense_matrix(M, N, density, A_dense);

  std::vector<double> B_dense;
  generate_dense_matrix(N, K, density, B_dense);

  std::vector<double> A_values;
  std::vector<int> A_col;
  std::vector<int> A_row;
  dense_to_crs(A_dense, M, N, A_values, A_col, A_row);

  std::vector<double> B_values;
  std::vector<int> B_col;
  std::vector<int> B_row;
  dense_to_crs(B_dense, N, K, B_values, B_col, B_row);

  size_t max_nnz = (size_t)M * (size_t)K;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs = {reinterpret_cast<uint8_t*>(A_values.data()), reinterpret_cast<uint8_t*>(A_col.data()),
                      reinterpret_cast<uint8_t*>(A_row.data()),    reinterpret_cast<uint8_t*>(B_values.data()),
                      reinterpret_cast<uint8_t*>(B_col.data()),    reinterpret_cast<uint8_t*>(B_row.data())};
  taskData->inputs_count = {(unsigned int)A_values.size(), (unsigned int)A_col.size(), (unsigned int)A_row.size(),
                            (unsigned int)B_values.size(), (unsigned int)B_col.size(), (unsigned int)B_row.size()};

  std::vector<double> C_val(max_nnz, 0.0);
  std::vector<int> C_col_index(max_nnz, 0);
  std::vector<int> C_row_ptr(M + 1, 0);

  taskData->outputs = {reinterpret_cast<uint8_t*>(C_val.data()), reinterpret_cast<uint8_t*>(C_col_index.data()),
                       reinterpret_cast<uint8_t*>(C_row_ptr.data())};
  taskData->outputs_count = {(unsigned int)C_val.size(), (unsigned int)C_col_index.size(),
                             (unsigned int)C_row_ptr.size()};

  borisov_s_crs_mul::CrsMatrixMulTask task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}
