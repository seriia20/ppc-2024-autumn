// Golovkin Maksim Task#1

#pragma once

#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace golovkin_integration_rectangular_method {

class IntegralCalculator : public ppc::core::Task {
 public:
  explicit IntegralCalculator(const std::shared_ptr<ppc::core::TaskData>& taskData);

  bool validation() override;
  bool pre_processing() override;
  bool post_processing() override;
  bool run() override;
  static double function_square(double x);

 private:
  std::shared_ptr<ppc::core::TaskData> taskData;
  double a;
  double b;
  double epsilon;
  int cnt_of_splits;
  double h;
  double res;
  std::vector<double> input_;
};

}  // namespace golovkin_integration_rectangular_method