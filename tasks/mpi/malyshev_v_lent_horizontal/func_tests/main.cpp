#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <climits>
#include <random>
#include <vector>

#include "mpi/malyshev_v_lent_horizontal/include/ops_mpi.hpp"

namespace malyshev_lent_horizontal {

std::vector<std::vector<int32_t>> generateRandomMatrix(uint32_t rows, uint32_t cols, int32_t min_value,
                                                       int32_t max_value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<std::vector<int32_t>> data(rows, std::vector<int32_t>(cols));

  for (auto &row : data) {
    for (auto &el : row) {
      el = min_value + gen() % (max_value - min_value + 1);
    }
  }

  return data;
}

std::vector<int32_t> generateRandomVector(uint32_t size, int32_t min_value, int32_t max_value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int32_t> data(size);

  for (auto &el : data) {
    el = min_value + gen() % (max_value - min_value + 1);
  }

  return data;
}

}  // namespace malyshev_lent_horizontal

TEST(malyshev_lent_horizontal, test_vertical_stretched_matrix_100x75) {
  uint32_t rows = 100;
  uint32_t cols = 75;
  int32_t min_value = -200;
  int32_t max_value = 300;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols, min_value, max_value);
    randomVector = malyshev_lent_horizontal::generateRandomVector(cols, min_value, max_value);
    mpiResult.resize(rows);

    for (auto &row : randomMatrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(rows);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<int32_t> seqResult(rows);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_lent_horizontal::TestTaskSequential taskSeq(taskDataSeq);

    for (auto &row : randomMatrix) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataSeq->inputs_count.push_back(rows);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.push_back(seqResult.size());

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < mpiResult.size(); i++) {
      ASSERT_EQ(seqResult[i], mpiResult[i]);
    }
  }
}

TEST(malyshev_lent_horizontal, test_horizontal_stretched_matrix_7x17) {
  uint32_t rows = 7;
  uint32_t cols = 17;
  int32_t min_value = -200;
  int32_t max_value = 300;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols, min_value, max_value);
    randomVector = malyshev_lent_horizontal::generateRandomVector(cols, min_value, max_value);
    mpiResult.resize(rows);

    for (auto &row : randomMatrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(rows);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<int32_t> seqResult(rows);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_lent_horizontal::TestTaskSequential taskSeq(taskDataSeq);

    for (auto &row : randomMatrix) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataSeq->inputs_count.push_back(rows);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.push_back(seqResult.size());

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < mpiResult.size(); i++) {
      ASSERT_EQ(seqResult[i], mpiResult[i]);
    }
  }
}

TEST(malyshev_lent_horizontal, test_square_matrix_100x100) {
  uint32_t rows = 100;
  uint32_t cols = 100;
  int32_t min_value = -200;
  int32_t max_value = 300;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols, min_value, max_value);
    randomVector = malyshev_lent_horizontal::generateRandomVector(cols, min_value, max_value);
    mpiResult.resize(rows);

    for (auto &row : randomMatrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(rows);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<int32_t> seqResult(rows);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_lent_horizontal::TestTaskSequential taskSeq(taskDataSeq);

    for (auto &row : randomMatrix) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataSeq->inputs_count.push_back(rows);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.push_back(seqResult.size());

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < mpiResult.size(); i++) {
      ASSERT_EQ(seqResult[i], mpiResult[i]);
    }
  }
}

TEST(malyshev_lent_horizontal, test_single_element_matrix_1x1) {
  uint32_t rows = 1;
  uint32_t cols = 1;
  int32_t min_value = -200;
  int32_t max_value = 300;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols, min_value, max_value);
    randomVector = malyshev_lent_horizontal::generateRandomVector(cols, min_value, max_value);
    mpiResult.resize(rows);

    for (auto &row : randomMatrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(rows);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<int32_t> seqResult(rows);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_lent_horizontal::TestTaskSequential taskSeq(taskDataSeq);

    for (auto &row : randomMatrix) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataSeq->inputs_count.push_back(rows);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.push_back(seqResult.size());

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < mpiResult.size(); i++) {
      ASSERT_EQ(seqResult[i], mpiResult[i]);
    }
  }
}

TEST(malyshev_lent_horizontal, test_validation_failure) {
  uint32_t rows = 7;
  uint32_t cols = 17;
  int32_t min_value = -200;
  int32_t max_value = 300;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols, min_value, max_value);
    randomVector = malyshev_lent_horizontal::generateRandomVector(cols, min_value, max_value);
    mpiResult.resize(rows);

    for (auto &row : randomMatrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->inputs_count.push_back(0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(0);

    ASSERT_FALSE(taskMPI.validation());
  }
}

TEST(malyshev_lent_horizontal, test_size_mismatch) {
  uint32_t rows = 3;
  uint32_t cols = 3;
  int32_t min_value = -200;
  int32_t max_value = 300;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols, min_value, max_value);
    randomVector = malyshev_lent_horizontal::generateRandomVector(cols + 1, min_value, max_value);
    mpiResult.resize(rows);

    for (auto &row : randomMatrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->inputs_count.push_back(cols + 1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(rows);

    ASSERT_FALSE(taskMPI.validation());
  }
}

TEST(malyshev_lent_horizontal, test_zero_values_mpi) {
  uint32_t rows = 3;
  uint32_t cols = 3;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> zeroMatrix(rows, std::vector<int32_t>(cols, 0));
  std::vector<int32_t> zeroVector(cols, 0);
  std::vector<int32_t> mpiResult(rows, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    for (auto &row : zeroMatrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(zeroVector.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(mpiResult.size());
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    for (uint32_t i = 0; i < mpiResult.size(); i++) {
      ASSERT_EQ(mpiResult[i], 0);
    }
  }
}