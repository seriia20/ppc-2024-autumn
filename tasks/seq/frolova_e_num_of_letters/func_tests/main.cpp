// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/frolova_e_num_of_letters/include/ops_seq.hpp"

std::string GenStr(int n) {
  if (n <= 0) {
    return std::string();
  }
  std::string str = "test";
  std::string result;
  result.resize(n);

  int i = 0;
  size_t j = 0;

  while (i < n) {
    result[i] = str[j];
    j++;
    i++;
    if (j >= str.size()) {
      j = 0;
    }
  }
  return result;
}

TEST(frolova_e_num_of_letters_seq, returns_empty_str_) {
  std::string str = GenStr(-2);
  EXPECT_TRUE(str.empty());
  std::string str2 = GenStr(0);
  EXPECT_TRUE(str2.empty());
}

TEST(frolova_e_num_of_letters_seq, returns__str_) {
  std::string str = GenStr(2);
  unsigned long size = 2;
  ASSERT_EQ(str.size(), size);
}

TEST(frolova_e_num_of_letters_seq, empty_str_test) {
  std::string str;

  // Create data
  std::vector<std::string> in(1, str);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataSeq->inputs_count.emplace_back(str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  frolova_e_num_of_letters_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(frolova_e_num_of_letters_seq, str_without_letters_test) {
  std::string str = "0";

  // Create data
  std::vector<std::string> in(1, str);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataSeq->inputs_count.emplace_back(str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  frolova_e_num_of_letters_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(0, out[0]);
}

TEST(frolova_e_num_of_letters_seq, str_with_one_letter_test) {
  std::string str = "a";

  // Create data
  std::vector<std::string> in(1, str);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataSeq->inputs_count.emplace_back(str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  frolova_e_num_of_letters_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(1, out[0]);
}

TEST(frolova_e_num_of_letters_seq, str_with_letters_test) {
  std::string str = "test";

  // Create data
  std::vector<std::string> in(1, str);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataSeq->inputs_count.emplace_back(str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  frolova_e_num_of_letters_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(4, out[0]);
}

TEST(frolova_e_num_of_letters_seq, str_with_letters_and_other_symbols_test) {
  std::string str = "123test;";

  // Create data
  std::vector<std::string> in(1, str);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataSeq->inputs_count.emplace_back(str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  frolova_e_num_of_letters_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(4, out[0]);
}