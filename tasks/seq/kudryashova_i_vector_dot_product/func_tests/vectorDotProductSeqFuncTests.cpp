#include <gtest/gtest.h>

#include "seq/kudryashova_i_vector_dot_product/include/vectorDotProductSeq.hpp"

static int seedOffset = 0;
std::vector<int> GetRandomVector(int size) {
  std::vector<int> vector(size);
  std::srand(static_cast<unsigned>(time(nullptr)) + ++seedOffset);
  for (int i = 0; i < size; ++i) {
    vector[i] = std::rand() % 100 + 1;
  }
  return vector;
}

TEST(kudryashova_i_vector_dot_product_seq, check_vector_dot_product_true) {
  // Create data
  std::vector<int> vector1 = {1, 8, 14};
  std::vector<int> vector2 = {3, 6, 5};
  ASSERT_EQ(121, kudryashova_i_vector_dot_product::vectorDotProduct(vector1, vector2));
}

TEST(kudryashova_i_vector_dot_product_seq, dot_product_vector_size_1) {
  const int count = 1;
  // Create data
  std::vector<int> vector1 = GetRandomVector(count);
  std::vector<int> vector2 = GetRandomVector(count);
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector1.data()));
  taskDataSeq->inputs_count.emplace_back(vector1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector2.data()));
  taskDataSeq->inputs_count.emplace_back(vector2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  kudryashova_i_vector_dot_product::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(kudryashova_i_vector_dot_product::vectorDotProduct(vector1, vector2), out[0]);
}

TEST(kudryashova_i_vector_dot_product_seq, dot_product_vector_size_50) {
  const int count = 50;
  // Create data
  std::vector<int> vector1 = GetRandomVector(count);
  std::vector<int> vector2 = GetRandomVector(count);
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector1.data()));
  taskDataSeq->inputs_count.emplace_back(vector1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector2.data()));
  taskDataSeq->inputs_count.emplace_back(vector2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  kudryashova_i_vector_dot_product::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(kudryashova_i_vector_dot_product::vectorDotProduct(vector1, vector2), out[0]);
}

TEST(kudryashova_i_vector_dot_product_seq, dot_product_vector_size_120) {
  const int count = 120;
  // Create data
  std::vector<int> out(1, 0);
  std::vector<int> vector1 = GetRandomVector(count);
  std::vector<int> vector2 = GetRandomVector(count);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector2.data()));
  taskDataSeq->inputs_count.emplace_back(vector1.size());
  taskDataSeq->inputs_count.emplace_back(vector2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  kudryashova_i_vector_dot_product::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(kudryashova_i_vector_dot_product::vectorDotProduct(vector1, vector2), out[0]);
}

TEST(kudryashova_i_vector_dot_product_seq, check_equal_vectors) {
  const int count = 10;
  // Create data
  std::vector<int> out(1, 0);
  std::vector<int> vector1 = GetRandomVector(count);
  std::vector<int> vector2 = GetRandomVector(count);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector2.data()));
  taskDataSeq->inputs_count.emplace_back(vector1.size());
  taskDataSeq->inputs_count.emplace_back(vector2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  kudryashova_i_vector_dot_product::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
}

TEST(kudryashova_i_vector_dot_product_seq, checks_not_equal_vector) {
  const int count = 10;
  // Create data
  std::vector<int> vector1 = GetRandomVector(count);
  std::vector<int> vector2 = GetRandomVector(count + 1);
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector2.data()));
  taskDataSeq->inputs_count.emplace_back(vector1.size());
  taskDataSeq->inputs_count.emplace_back(vector2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  kudryashova_i_vector_dot_product::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(kudryashova_i_vector_dot_product_seq, check_empty_vectors) {
  // Create data
  std::vector<int> vector1 = {};
  std::vector<int> vector2 = {};
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector2.data()));
  taskDataSeq->inputs_count.emplace_back(vector1.size());
  taskDataSeq->inputs_count.emplace_back(vector2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  kudryashova_i_vector_dot_product::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(kudryashova_i_vector_dot_product_seq, check_run_true) {
  // Create data
  std::vector<int> out(1, 0);
  std::vector<int> vector1 = {1, 8, 14};
  std::vector<int> vector2 = {3, 6, 5};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector2.data()));
  taskDataSeq->inputs_count.emplace_back(vector1.size());
  taskDataSeq->inputs_count.emplace_back(vector2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  kudryashova_i_vector_dot_product::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(121, out[0]);
}