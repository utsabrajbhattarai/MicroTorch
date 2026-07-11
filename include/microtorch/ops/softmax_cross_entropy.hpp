#pragma once
#include "microtorch/Tensor.hpp"

namespace microtorch {

// fused softmax + cross-entropy: combination for clean backward and together forms a clean classification loss function

TensorPtr softmax_cross_entropy(const TensorPtr& logits, const Eigen::MatrixXd& targets);

}  // namespace microtorch
