#pragma once
#include "microtorch/Tensor.hpp"

namespace microtorch {

// sum/mean reduction -- shared primitive every loss needs
// TODO: declare forward + backward signatures once Tensor's design is set.
Tensor sum_mean(const Tensor& a, const Tensor& b);

}  // namespace microtorch
