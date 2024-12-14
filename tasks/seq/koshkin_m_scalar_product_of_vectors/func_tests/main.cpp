// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>

#include "seq/koshkin_m_scalar_product_of_vectors/include/ops_seq.hpp"

static int offset = 0;

int koshkin_m_scalar_product_of_vectors::calculateDotProduct(const std::vector<int> &vec_1,
                                                             const std::vector<int> &vec_2) {
  long result = 0;
  for (size_t i = 0; i < vec_1.size(); i++) result += vec_1[i] * vec_2[i];
  return result;
}

int koshkin_m_scalar_product_of_vectors::generateRandomNumber(int min, int max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(min, max);
  return distrib(gen);
}

std::vector<int> koshkin_m_scalar_product_of_vectors::generateRandomVector(int v_size) {
  std::vector<int> vec(v_size);
  std::mt19937 gen;
  gen.seed((unsigned)time(nullptr) + ++offset);
  for (int i = 0; i < v_size; i++) vec[i] = gen() % 100;
  return vec;
}

TEST(koshkin_m_scalar_product_of_vectors, check_empty_func_vector_product) {
  const int count = 0;
  std::vector<int> vec_1 = koshkin_m_scalar_product_of_vectors::generateRandomVector(count);
  std::vector<int> vec_2 = koshkin_m_scalar_product_of_vectors::generateRandomVector(count);
  int answer = koshkin_m_scalar_product_of_vectors::calculateDotProduct(vec_1, vec_2);
  ASSERT_EQ(0, answer);
}

TEST(koshkin_m_scalar_product_of_vectors, check_scalary_product_of_vectors_size_20) {
  const int count = 20;

  std::vector<int> out(1, 0);
  std::vector<int> vec_1 = koshkin_m_scalar_product_of_vectors::generateRandomVector(count);
  std::vector<int> vec_2 = koshkin_m_scalar_product_of_vectors::generateRandomVector(count);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec_1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec_2.data()));

  taskDataSeq->inputs_count.emplace_back(vec_1.size());
  taskDataSeq->inputs_count.emplace_back(vec_2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  koshkin_m_scalar_product_of_vectors::VectorDotProduct VectorDotProduct(taskDataSeq);
  ASSERT_EQ(VectorDotProduct.validation(), true);
  VectorDotProduct.pre_processing();
  VectorDotProduct.run();
  VectorDotProduct.post_processing();
  int answer = koshkin_m_scalar_product_of_vectors::calculateDotProduct(vec_1, vec_2);
  ASSERT_EQ(answer, out[0]);
}

TEST(koshkin_m_scalar_product_of_vectors, check_scalary_product_of_vectors_size_300) {
  const int count = 300;

  std::vector<int> out(1, 0);
  std::vector<int> vec_1 = koshkin_m_scalar_product_of_vectors::generateRandomVector(count);
  std::vector<int> vec_2 = koshkin_m_scalar_product_of_vectors::generateRandomVector(count);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec_1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec_2.data()));

  taskDataSeq->inputs_count.emplace_back(vec_1.size());
  taskDataSeq->inputs_count.emplace_back(vec_2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  koshkin_m_scalar_product_of_vectors::VectorDotProduct VectorDotProduct(taskDataSeq);
  ASSERT_EQ(VectorDotProduct.validation(), true);
  VectorDotProduct.pre_processing();
  VectorDotProduct.run();
  VectorDotProduct.post_processing();
  int answer = koshkin_m_scalar_product_of_vectors::calculateDotProduct(vec_1, vec_2);
  ASSERT_EQ(answer, out[0]);
}

TEST(koshkin_m_scalar_product_of_vectors, check_run_correct_binary_sys) {
  std::vector<int> out(1, 0);

  std::vector<int> vec_1 = {5, 2};
  std::vector<int> vec_2 = {6, 10};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec_1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec_2.data()));

  taskDataSeq->inputs_count.emplace_back(vec_1.size());
  taskDataSeq->inputs_count.emplace_back(vec_2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  koshkin_m_scalar_product_of_vectors::VectorDotProduct VectorDotProduct(taskDataSeq);
  ASSERT_EQ(VectorDotProduct.validation(), true);
  VectorDotProduct.pre_processing();
  VectorDotProduct.run();
  VectorDotProduct.post_processing();
  ASSERT_EQ(50, out[0]);
}

TEST(koshkin_m_scalar_product_of_vectors, check_calculateDotProduct_right_binary_sys) {
  std::vector<int> vec_1 = {5, 2};
  std::vector<int> vec_2 = {7, 3};
  ASSERT_EQ(41, koshkin_m_scalar_product_of_vectors::calculateDotProduct(vec_1, vec_2));
}

TEST(koshkin_m_scalar_product_of_vectors, check_calculateDotProduct_right_ternary_sys) {
  std::vector<int> vec_1 = {5, 2, 10};
  std::vector<int> vec_2 = {7, 3, 1};
  ASSERT_EQ(51, koshkin_m_scalar_product_of_vectors::calculateDotProduct(vec_1, vec_2));
}

TEST(koshkin_m_scalar_product_of_vectors, check_calculateDotProduct_binary_neg) {
  std::vector<int> vec_1 = {-1, -8};
  std::vector<int> vec_2 = {-5, 7};
  ASSERT_EQ(-51, koshkin_m_scalar_product_of_vectors::calculateDotProduct(vec_1, vec_2));
}

TEST(koshkin_m_scalar_product_of_vectors, check_calculateDotProduct_random_size) {
  int size = koshkin_m_scalar_product_of_vectors::generateRandomNumber(1, 100);
  std::vector<int> vec_1(size);
  std::vector<int> vec_2(size);
  for (int i = 0; i < size; ++i) {
    vec_1[i] = koshkin_m_scalar_product_of_vectors::generateRandomNumber(-10, 10);
    vec_2[i] = koshkin_m_scalar_product_of_vectors::generateRandomNumber(-10, 10);
  }

  long expected_result = 0;
  for (int i = 0; i < size; ++i) {
    expected_result += vec_1[i] * vec_2[i];
  }

  ASSERT_EQ(expected_result, koshkin_m_scalar_product_of_vectors::calculateDotProduct(vec_1, vec_2));
}