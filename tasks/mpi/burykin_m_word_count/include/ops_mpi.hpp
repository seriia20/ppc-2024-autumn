#pragma once

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace burykin_m_word_count {

std::vector<char> RandomSentence(int size);

// Последовательная версия
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_;
  int word_count_{};

  static bool is_word_character(char c);
  static int count_words(const std::string& text);
};

// Параллельная версия
class TestTaskParallel : public ppc::core::Task {
 public:
  explicit TestTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  std::vector<char> input_;
  std::vector<char> local_input_;
  int word_count_{};
  int local_word_count_{};
  int length{};

  static bool is_word_character(char c);
  int count_words(const std::vector<char>& text);
};

}  // namespace burykin_m_word_count
