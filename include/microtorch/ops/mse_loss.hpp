#pragma once
#include "microtorch/Tensor.hpp"

namespace microtorch {

// MSE loss: diffusion's training objective

TensorPtr mse_loss(const Tensor& pred, const Eigen::MatrixXd& target);

}  // namespace microtorch
