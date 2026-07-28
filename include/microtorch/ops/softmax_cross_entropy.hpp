#pragma once
#include "microtorch/Tensor.hpp"

namespace microtorch {

// fused softmax + cross-entropy: combination for clean backward and together forms a clean classification loss function

//overloaded to ensure masked and unmasked can be applied to a same function: more of it in actual src/*.cpp file

TensorPtr softmax_cross_entropy(const TensorPtr& logits, const Eigen::MatrixXd& targets);
TensorPtr softmax_cross_entropy(const TensorPtr& logits, const Eigen::MatrixXd& targets, const Eigen::MatrixXd& mask, double num_labeled); //masked version for loss with no classes

}  // namespace microtorch
