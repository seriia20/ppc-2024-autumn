// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "seq/suvorov_d_sum_of_vector_elements/include/vec.hpp"

TEST(suvorov_d_sum_of_vector_elements_seq, Test_Sum_1000) {
  // Create data
  const size_t vec_size = 100;
  std::vector<int> input_test_vector(vec_size);
  std::vector<int> test_output(1, 0);

  // Initialize an input vector with random integers and getting the correct sum result
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(-100, 100);
  int right_result = 0;
  for (size_t i = 0; i < vec_size; ++i) {
    input_test_vector[i] = dis(gen);
    right_result += input_test_vector[i];
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_test_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_test_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(test_output.data()));
  taskDataSeq->outputs_count.emplace_back(test_output.size());

  // Create Task
  suvorov_d_sum_of_vector_elements_seq::Sum_of_vector_elements_seq SumOfVectorElementsSeq(taskDataSeq);
  ASSERT_EQ(SumOfVectorElementsSeq.validation(), true);
  SumOfVectorElementsSeq.pre_processing();
  SumOfVectorElementsSeq.run();
  SumOfVectorElementsSeq.post_processing();
  ASSERT_EQ(right_result, test_output[0]);
}

TEST(suvorov_d_sum_of_vector_elements_seq, Test_Sum_10000000) {
  // Create data
  const size_t vec_size = 10000000;
  std::vector<int> input_test_vector(vec_size);
  std::vector<int> test_output(1, 0);

  // Initialize an input vector with random integers and getting the correct sum result
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(-1000000, 1000000);
  int right_result = 0;
  for (size_t i = 0; i < vec_size; ++i) {
    input_test_vector[i] = dis(gen);
    right_result += input_test_vector[i];
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_test_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_test_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(test_output.data()));
  taskDataSeq->outputs_count.emplace_back(test_output.size());

  // Create Task
  suvorov_d_sum_of_vector_elements_seq::Sum_of_vector_elements_seq SumOfVectorElementsSeq(taskDataSeq);
  ASSERT_EQ(SumOfVectorElementsSeq.validation(), true);
  SumOfVectorElementsSeq.pre_processing();
  SumOfVectorElementsSeq.run();
  SumOfVectorElementsSeq.post_processing();
  ASSERT_EQ(right_result, test_output[0]);
}

TEST(suvorov_d_sum_of_vector_elements_seq, Test_Sum_With_Single_Element) {
  // Create data
  const size_t vec_size = 100;
  std::vector<int> input_test_vector(vec_size);
  std::vector<int> test_output(1, 0);

  // Initialize an input vector with random integers and getting the correct sum result
  int right_result = 0;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(-1000000, 1000000);
  input_test_vector[0] = dis(gen);
  right_result = input_test_vector[0];

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_test_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_test_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(test_output.data()));
  taskDataSeq->outputs_count.emplace_back(test_output.size());

  // Create Task
  suvorov_d_sum_of_vector_elements_seq::Sum_of_vector_elements_seq SumOfVectorElementsSeq(taskDataSeq);
  ASSERT_EQ(SumOfVectorElementsSeq.validation(), true);
  SumOfVectorElementsSeq.pre_processing();
  SumOfVectorElementsSeq.run();
  SumOfVectorElementsSeq.post_processing();
  ASSERT_EQ(right_result, test_output[0]);
}

TEST(suvorov_d_sum_of_vector_elements_seq, Test_Sum_With_Zero_Vector) {
  // Create data
  const size_t vec_size = 100;
  std::vector<int> input_test_vector(vec_size, 0);
  std::vector<int> test_output(1, 0);

  // Initialize an input vector with random integers and getting the correct sum result
  int right_result = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_test_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_test_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(test_output.data()));
  taskDataSeq->outputs_count.emplace_back(test_output.size());

  // Create Task
  suvorov_d_sum_of_vector_elements_seq::Sum_of_vector_elements_seq SumOfVectorElementsSeq(taskDataSeq);
  ASSERT_EQ(SumOfVectorElementsSeq.validation(), true);
  SumOfVectorElementsSeq.pre_processing();
  SumOfVectorElementsSeq.run();
  SumOfVectorElementsSeq.post_processing();
  ASSERT_EQ(right_result, test_output[0]);
}

TEST(suvorov_d_sum_of_vector_elements_seq, Test_Sum_With_Empty_Vector) {
  // Create data
  std::vector<int> input_test_vector;
  std::vector<int> test_output(1, 0);

  // Initialize an input vector with random integers and getting the correct sum result
  int right_result = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_test_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_test_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(test_output.data()));
  taskDataSeq->outputs_count.emplace_back(test_output.size());

  // Create Task
  suvorov_d_sum_of_vector_elements_seq::Sum_of_vector_elements_seq SumOfVectorElementsSeq(taskDataSeq);
  ASSERT_EQ(SumOfVectorElementsSeq.validation(), true);
  SumOfVectorElementsSeq.pre_processing();
  SumOfVectorElementsSeq.run();
  SumOfVectorElementsSeq.post_processing();
  ASSERT_EQ(right_result, test_output[0]);
}
