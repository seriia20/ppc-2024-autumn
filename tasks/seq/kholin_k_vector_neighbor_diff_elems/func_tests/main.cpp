#include <gtest/gtest.h>

#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kholin_k_vector_neighbor_diff_elems/include/ops_seq.hpp"

TEST(kholin_k_vector_neighbor_diff_elems_seq, check_pre_processing) {
  std::vector<int32_t> in(1256, 1);
  std::vector<int32_t> out(2, 0);
  std::vector<uint64_t> out_index(2, 0);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  kholin_k_vector_neighbor_diff_elems_seq::TestTaskSequential<int32_t, uint64_t> testTaskSequential(taskData);
  testTaskSequential.validation();

  EXPECT_EQ(testTaskSequential.pre_processing(), true);
}

TEST(kholin_k_vector_neighbor_diff_elems_seq, check_validation) {
  std::vector<int32_t> in(1256, 1);
  std::vector<int32_t> out(2, 0);
  std::vector<uint64_t> out_index(2, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  kholin_k_vector_neighbor_diff_elems_seq::TestTaskSequential<int32_t, uint64_t> testTaskSequential(taskData);
  EXPECT_EQ(testTaskSequential.validation(), true);
}

TEST(kholin_k_vector_neighbor_diff_elems_seq, check_run) {
  std::vector<int32_t> in(1256, 1);
  std::vector<int32_t> out(2, 0);
  std::vector<uint64_t> out_index(2, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  kholin_k_vector_neighbor_diff_elems_seq::TestTaskSequential<int32_t, uint64_t> testTaskSequential(taskData);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  EXPECT_EQ(testTaskSequential.run(), true);
}

TEST(kholin_k_vector_neighbor_diff_elems_seq, check_post_processing) {
  std::vector<int32_t> in(1256, 1);
  std::vector<int32_t> out(2, 0);
  std::vector<uint64_t> out_index(2, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  kholin_k_vector_neighbor_diff_elems_seq::TestTaskSequential<int32_t, uint64_t> testTaskSequential(taskData);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  EXPECT_EQ(testTaskSequential.post_processing(), true);
}

TEST(kholin_k_vector_neighbor_diff_elems_seq, check_int32_t) {
  std::vector<int32_t> in(1256, 1);
  std::vector<int32_t> out(2, 0);
  std::vector<uint64_t> out_index(2, 0);
  for (size_t i = 0; i < in.size(); i++) {
    in[i] = 2 * i;
  }
  in[234] = 0;
  in[235] = 4000;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  kholin_k_vector_neighbor_diff_elems_seq::TestTaskSequential<int32_t, uint64_t> testTaskSequential(taskData);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_EQ(out[0], 0l);
  EXPECT_EQ(out[1], 4000l);
  EXPECT_EQ(out_index[0], 234ull);
  EXPECT_EQ(out_index[1], 235ull);
}

TEST(kholin_k_vector_neighbor_diff_elems_seq, check_int_with_random) {
  std::vector<int> in(1256, 1);
  std::vector<int> out(2, 0);
  std::vector<uint64_t> out_index(2, 0);
  in = kholin_k_vector_neighbor_diff_elems_seq::get_random_vector<int>(1256);
  in[234] = 0;
  in[235] = 4000;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  kholin_k_vector_neighbor_diff_elems_seq::TestTaskSequential<int, uint64_t> testTaskSequential(taskData);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_EQ(out[0], 0l);
  EXPECT_EQ(out[1], 4000l);
  EXPECT_EQ(out_index[0], 234ull);
  EXPECT_EQ(out_index[1], 235ull);
}

TEST(kholin_k_vector_neighbour_diff_elems_seq, check_double) {
  std::vector<double> in(25680, 1);
  std::vector<double> out(2, 0);
  std::vector<uint64_t> out_index(2, 0);
  for (size_t i = 0; i < in.size(); i++) {
    in[i] = i;
  }
  in[189] = -1000.1;
  in[190] = 9000.9;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  kholin_k_vector_neighbor_diff_elems_seq::TestTaskSequential<double, uint64_t> testTaskSequential(taskData);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_NEAR(out[0], -1000.1, 1e-6);
  EXPECT_NEAR(out[1], 9000.9, 1e-6);
  EXPECT_EQ(out_index[0], 189ull);
  EXPECT_EQ(out_index[1], 190ull);
}

TEST(kholin_k_vector_neighbour_diff_elems_seq, check_int8_t) {
  std::vector<int8_t> in(250, -1);
  std::vector<int8_t> out(2, 0);
  std::vector<uint64_t> out_index(2, 0);
  for (size_t i = 0; i < in.size(); i++) {
    if (i % 2 == 0) {
      in[i] = -50;
    } else {
      in[i] = 50;
    }
  }
  in[5] = 56;
  in[6] = -56;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  kholin_k_vector_neighbor_diff_elems_seq::TestTaskSequential<int8_t, uint64_t> testTaskSequential(taskData);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_EQ(out[0], 56);
  EXPECT_EQ(out[1], -56);
  EXPECT_EQ(out_index[0], 5ull);
  EXPECT_EQ(out_index[1], 6ull);
}

TEST(kholin_k_vector_neighbour_diff_elems_seq, check_int64_t) {
  std::vector<int64_t> in(75836, 1);
  std::vector<int64_t> out(2, 0);
  std::vector<uint64_t> out_index(2, 0);
  for (size_t i = 0; i < in.size(); i++) {
    if (i % 3 == 0) {
      in[i] = 10;
    }
    if (i % 3 == 1) {
      in[i] = 30;
    }
    if (i % 3 == 2) {
      in[i] = 70;
    }
  }
  in[20] = -1000;
  in[21] = 1119;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  kholin_k_vector_neighbor_diff_elems_seq::TestTaskSequential<int64_t, uint64_t> testTaskSequential(taskData);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_EQ(out[0], -1000ll);
  EXPECT_EQ(out[1], 1119ll);
  EXPECT_EQ(out_index[0], 20ull);
  EXPECT_EQ(out_index[1], 21ull);
}

TEST(kholin_k_vector_neighbour_diff_elems_seq, check_float) {
  std::vector<float> in(20, 1.0f);
  std::vector<float> out(2, 0.0f);
  std::vector<uint64_t> out_index(2, 0);
  for (size_t i = 0; i < in.size(); i++) {
    in[i] += (i + 1.0f) * 2.5f;
  }
  in[0] = 110.001f;
  in[1] = -990.0025f;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  kholin_k_vector_neighbor_diff_elems_seq::TestTaskSequential<float, uint64_t> testTaskSequential(taskData);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_NEAR(out[0], 110.001f, 1e-4);
  EXPECT_NEAR(out[1], -990.0025f, 1e-4);
  EXPECT_EQ(out_index[0], 0ull);
  EXPECT_EQ(out_index[1], 1ull);
}

TEST(kholin_k_vector_neighbour_diff_elems_seq, check_float_with_random) {
  std::vector<float> in(20, 1.0f);
  std::vector<float> out(2, 0.0f);
  std::vector<uint64_t> out_index(2, 0);
  in = kholin_k_vector_neighbor_diff_elems_seq::get_random_vector<float>(20);
  in[0] = 110.001f;
  in[1] = -990.0025f;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  kholin_k_vector_neighbor_diff_elems_seq::TestTaskSequential<float, uint64_t> testTaskSequential(taskData);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_NEAR(out[0], 110.001f, 1e-4);
  EXPECT_NEAR(out[1], -990.0025f, 1e-4);
  EXPECT_EQ(out_index[0], 0ull);
  EXPECT_EQ(out_index[1], 1ull);
}