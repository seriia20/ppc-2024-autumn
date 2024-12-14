// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "seq/yasakova_t_min_of_vector_elements/include/ops_seq_yasakova.hpp"

std::vector<int> RandomVector(int size, int minimum = 0, int maximum = 100) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = minimum + gen() % (maximum - minimum + 1);
  }
  return vec;
}

std::vector<std::vector<int>> RandomMatrix(int rows, int columns, int minimum = 0, int maximum = 100) {
  std::vector<std::vector<int>> vec(rows);
  for (int i = 0; i < rows; i++) {
    vec[i] = RandomVector(columns, minimum, maximum);
  }
  return vec;
}

TEST(yasakova_t_min_of_vector_elements_seq, testFindMinimumInMatrixWithOneRow) {
  std::random_device dev;
  std::mt19937 gen(dev());
  const int count_rows = 1;
  const int count_columns = 10;
  const int gen_minimum = -500;
  const int gen_maximum = 500;
  int ref = INT_MIN;
  std::vector<int> out(1, INT_MAX);
  std::vector<std::vector<int>> in = RandomMatrix(count_rows, count_columns, gen_minimum, gen_maximum);
  int index = gen() % count_columns;
  in[0][index] = ref;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  yasakova_t_min_of_vector_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out[0]);
}

TEST(yasakova_t_min_of_vector_elements_seq, testFindMinimumIn10x10Matrix) {
  std::random_device dev;
  std::mt19937 gen(dev());
  const int count_rows = 10;
  const int count_columns = 10;
  const int gen_minimum = -500;
  const int gen_maximum = 500;
  int ref = INT_MIN;
  std::vector<int> out(1, INT_MAX);
  std::vector<std::vector<int>> in = RandomMatrix(count_rows, count_columns, gen_minimum, gen_maximum);
  int index = gen() % (count_rows * count_columns);
  in[index / count_columns][index / count_rows] = ref;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  yasakova_t_min_of_vector_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out[0]);
}

TEST(yasakova_t_min_of_vector_elements_seq, testFindMinimumIn10x100Matrix) {
  std::random_device dev;
  std::mt19937 gen(dev());
  const int count_rows = 10;
  const int count_columns = 100;
  const int gen_minimum = -500;
  const int gen_maximum = 500;
  int ref = INT_MIN;
  std::vector<int> out(1, INT_MAX);
  std::vector<std::vector<int>> in = RandomMatrix(count_rows, count_columns, gen_minimum, gen_maximum);
  int index = gen() % (count_rows * count_columns);
  in[index / count_columns][index / count_rows] = ref;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  yasakova_t_min_of_vector_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out[0]);
}

TEST(yasakova_t_min_of_vector_elements_seq, testFindMinimumIn100x10Matrix) {
  std::random_device dev;
  std::mt19937 gen(dev());
  const int count_rows = 100;
  const int count_columns = 10;
  const int gen_minimum = -500;
  const int gen_maximum = 500;
  int ref = INT_MIN;
  std::vector<int> out(1, INT_MAX);
  std::vector<std::vector<int>> in = RandomMatrix(count_rows, count_columns, gen_minimum, gen_maximum);
  int index = gen() % (count_rows * count_columns);
  in[index / count_columns][index / count_rows] = ref;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  yasakova_t_min_of_vector_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out[0]);
}

TEST(yasakova_t_min_of_vector_elements_seq, testFindMinimumIn100x100Matrix) {
  std::random_device dev;
  std::mt19937 gen(dev());
  const int count_rows = 100;
  const int count_columns = 100;
  const int gen_minimum = -500;
  const int gen_maximum = 500;
  int ref = INT_MIN;
  std::vector<int> out(1, INT_MAX);
  std::vector<std::vector<int>> in = RandomMatrix(count_rows, count_columns, gen_minimum, gen_maximum);
  int index = gen() % (count_rows * count_columns);
  in[index / count_columns][index / count_rows] = ref;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  yasakova_t_min_of_vector_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out[0]);
}