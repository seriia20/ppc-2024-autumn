// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/drozhdinov_d_gauss_vertical_scheme/include/ops_seq.hpp"

namespace drozhdinov_d_gauss_vertical_scheme_seq {
std::vector<double> genElementaryMatrix(int rows, int columns) {
  std::vector<double> res;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      if (i == j) {
        res.push_back(1);
      } else {
        res.push_back(0);
      }
    }
  }
  return res;
}

std::vector<double> genDenseMatrix(int n, int a) {
  std::vector<double> dense;
  std::vector<double> ed(n * n);
  std::vector<double> res(n * n);
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n + i; j++) {
      dense.push_back(a + j);
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i < 2) {
        ed[j * n + i] = 0;
      } else if (i == j && i >= 2) {
        ed[j * n + i] = 1;
      } else {
        ed[j * n + i] = 0;
      }
    }
  }
  for (int i = 0; i < n * n; i++) {
    res[i] = (dense[i] + ed[i]);
  }
  return res;
}

template <typename T>
std::vector<T> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<T> vec(sz);
  vec[0] = gen() % 100;
  for (int i = 1; i < sz; i++) {
    vec[i] = (gen() % 100) - 49;
  }
  return vec;
}

template std::vector<int> drozhdinov_d_gauss_vertical_scheme_seq::getRandomVector(int sz);
template std::vector<double> drozhdinov_d_gauss_vertical_scheme_seq::getRandomVector(int sz);
}  // namespace drozhdinov_d_gauss_vertical_scheme_seq

TEST(Sequential, EquationTest) {
  // Create data
  int rows = 2;
  int columns = 2;
  std::vector<double> matrix = {1, 0, 0, 1};
  std::vector<double> b = {1, 1};
  std::vector<double> expres(rows, 0);
  std::vector<double> res = {1, 1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, res);
}

TEST(Sequential, Equation2Test) {
  // Create data
  int rows = 3;
  int columns = 3;
  std::vector<double> matrix = {1, 2, 3, 2, 3, 4, 3, 4, 6};
  std::vector<double> b = {1, 1, 1};
  std::vector<double> expres(rows, 0);
  std::vector<double> res = {-1, 1, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, res);
}

TEST(Sequential, Equation3Test) {
  // Create data
  int rows = 3;
  int columns = 3;
  std::vector<double> matrix = drozhdinov_d_gauss_vertical_scheme_seq::genDenseMatrix(rows, 1);
  std::vector<double> b = {1, 1, 1};
  std::vector<double> expres(rows, 0);
  std::vector<double> res = {-1, 1, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, res);
}

TEST(Sequential, EmptyTest) {
  // Create data
  int rows = 0;
  int columns = 0;
  std::vector<double> matrix = {};
  std::vector<double> b = {};
  std::vector<double> expres(rows, 0);
  std::vector<double> res = {};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, res);
}

TEST(Sequential, Size100TestIdentity) {
  // Create data
  int rows = 10;
  int columns = 10;
  std::vector<double> matrix = drozhdinov_d_gauss_vertical_scheme_seq::genElementaryMatrix(rows, columns);
  std::vector<double> b(rows * columns, 1);
  std::vector<double> expres(rows);
  std::vector<double> res(rows, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, res);
}

TEST(Sequential, Size10000TestIdentity) {
  // Create data
  int rows = 100;
  int columns = 100;
  std::vector<double> matrix = drozhdinov_d_gauss_vertical_scheme_seq::genElementaryMatrix(rows, columns);
  std::vector<double> b(rows * columns, 1);
  std::vector<double> expres(rows);
  std::vector<double> res(rows, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, res);
}

TEST(Sequential, Size100TestDense) {
  // Create data
  int rows = 10;
  int columns = 10;
  std::vector<int> a = drozhdinov_d_gauss_vertical_scheme_seq::getRandomVector<int>(1);
  std::vector<double> matrix = drozhdinov_d_gauss_vertical_scheme_seq::genDenseMatrix(rows, *a.begin());
  std::vector<double> b(rows, 1);
  std::vector<double> expres(rows, 0);
  std::vector<double> res(rows, 0);
  res[0] = -1;
  res[1] = 1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, res);
}

TEST(Sequential, Size10000TestDense) {
  // Create data
  int rows = 100;
  int columns = 100;
  std::vector<int> a = drozhdinov_d_gauss_vertical_scheme_seq::getRandomVector<int>(1);
  std::vector<double> matrix = drozhdinov_d_gauss_vertical_scheme_seq::genDenseMatrix(rows, *a.begin());
  std::vector<double> b(rows, 1);
  std::vector<double> expres(rows, 0);
  std::vector<double> res(rows, 0);
  res[0] = -1;
  res[1] = 1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, res);
}

TEST(Sequential, WrongValidationTest1) {
  // Create data
  int rows = 20;
  int columns = 10;
  std::vector<double> matrix = drozhdinov_d_gauss_vertical_scheme_seq::genElementaryMatrix(rows, columns);
  std::vector<double> b(rows * columns, 1);
  std::vector<double> expres(rows);
  std::vector<double> res(rows, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(Sequential, WrongValidationTest2) {
  // Create data
  int rows = 20;
  int columns = 20;
  std::vector<double> matrix = drozhdinov_d_gauss_vertical_scheme_seq::genElementaryMatrix(rows, columns);
  std::vector<double> b(rows * columns, 1);
  std::vector<double> expres(rows + 5);
  std::vector<double> res(rows, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(Sequential, WrongPPTest) {
  // Create data
  int rows = 3;
  int columns = 3;
  std::vector<double> matrix = {1, 0, 0, 0, 1, 0, 0, 0, 0};  // det=0
  std::vector<double> b = {1, 1, 1};
  std::vector<double> expres(rows);
  std::vector<double> res(rows, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), false);
}