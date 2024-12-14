// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/morozov_e_writers_readers/include/ops_mpi.hpp"
TEST(morozov_e_writers_readers, Test_Validation_is_False1) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> vec{0};
  std::vector<int> ans{1};
  if (world.rank() == 0) {
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    data->inputs_count.emplace_back(0);
    data->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    data->outputs_count.emplace_back(ans.size());
    morozov_e_writers_readers::TestMPITaskParallel obj(data);
    ASSERT_FALSE(obj.validation());
  }
}
TEST(morozov_e_writers_readers, Test_Validation_is_False2) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> vec{0};
  std::vector<int> ans{1};
  if (world.rank() == 0) {
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    data->inputs_count.emplace_back(vec.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    data->outputs_count.emplace_back(0);
    morozov_e_writers_readers::TestMPITaskParallel obj(data);
    ASSERT_FALSE(obj.validation());
  }
}
TEST(morozov_e_writers_readers, Test_Validation_is_False3) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> vec{0};
  std::vector<int> ans{1};
  if (world.rank() == 0) {
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    data->outputs_count.emplace_back(0);
    morozov_e_writers_readers::TestMPITaskParallel obj(data);
    ASSERT_FALSE(obj.validation());
  }
}
TEST(morozov_e_writers_readers, Test_Validation_is_False4) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> vec{0};
  std::vector<int> ans{1};
  if (world.rank() == 0) {
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    data->inputs_count.emplace_back(vec.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    morozov_e_writers_readers::TestMPITaskParallel obj(data);
    ASSERT_FALSE(obj.validation());
  }
}
TEST(morozov_e_writers_readers, Test_Validation_isTrue) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> vec{0};
  std::vector<int> ans{1};
  if (world.rank() == 0) {
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    data->inputs_count.emplace_back(vec.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    data->outputs_count.emplace_back(ans.size());
    morozov_e_writers_readers::TestMPITaskParallel obj(data);
    ASSERT_TRUE(obj.validation());
  }
}
TEST(morozov_e_writers_readers, Test_Main0) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> vec{0};
  std::vector<int> ans{1};
  int countIteration = 10;
  int cur_value = 20;
  if (world.rank() == 0) {
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(new int{countIteration}));
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(new int{cur_value}));
    data->inputs_count.emplace_back(vec.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    data->outputs_count.emplace_back(ans.size());
  }
  morozov_e_writers_readers::TestMPITaskParallel obj(data);
  ASSERT_TRUE(obj.validation());
  obj.pre_processing();
  obj.run();
  obj.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(ans[0], 0);
  }
}
TEST(morozov_e_writers_readers, Test_Main1) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> vec{0};
  std::vector<int> ans{100};
  int countIteration = 100;
  int cur_value = 100;
  if (world.rank() == 0) {
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(new int{countIteration}));
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(new int{cur_value}));
    data->inputs_count.emplace_back(vec.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    data->outputs_count.emplace_back(ans.size());
  }
  morozov_e_writers_readers::TestMPITaskParallel obj(data);
  ASSERT_TRUE(obj.validation());
  obj.pre_processing();
  obj.run();
  obj.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(ans[0], 0);
  }
}
TEST(morozov_e_writers_readers, Test_Main2) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> vec{0};
  std::vector<int> ans{INT_MAX};
  int countIteration = 100;
  int cur_value = 1000;
  if (world.rank() == 0) {
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(new int{countIteration}));
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(new int{cur_value}));
    data->inputs_count.emplace_back(vec.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    data->outputs_count.emplace_back(ans.size());
  }
  morozov_e_writers_readers::TestMPITaskParallel obj(data);
  ASSERT_TRUE(obj.validation());
  obj.pre_processing();
  obj.run();
  obj.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(ans[0], 0);
  }
}

TEST(morozov_e_writers_readers, Test_Main3) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  std::vector<int> vec{0};
  std::vector<int> ans{-1};
  int countIteration = 200;
  int cur_value = -10;
  if (world.rank() == 0) {
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(new int{countIteration}));
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(new int{cur_value}));
    data->inputs_count.emplace_back(vec.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    data->outputs_count.emplace_back(ans.size());
  }
  morozov_e_writers_readers::TestMPITaskParallel obj(data);
  ASSERT_TRUE(obj.validation());
  obj.pre_processing();
  obj.run();
  obj.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(ans[0], 0);
  }
}
