#pragma once
#include "microtorch/Tensor.hpp"

namespace microtorch {

// fused softmax + cross-entropy -- GNN's training objective, fused for a clean O(n) backward
// TODO: declare forward + backward signatures once Tensor's design is set.
TensorPtr softmax_cross_entropy(const TensorPtr& logits, const Eigen::MatrixXd& targets);

}  // namespace microtorch
