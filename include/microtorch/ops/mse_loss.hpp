#pragma once
#include "microtorch/Tensor.hpp"

namespace microtorch {

// MSE loss -- diffusion's training objective
// TODO: declare forward + backward signatures once Tensor's design is set.
Tensor mse_loss(const Tensor& a, const Tensor& b);

}  // namespace microtorch
