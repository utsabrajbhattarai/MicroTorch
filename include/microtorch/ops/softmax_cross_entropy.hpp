#pragma once
#include "microtorch/Tensor.hpp"

namespace microtorch {

// fused softmax + cross-entropy -- GNN's training objective, fused for a clean O(n) backward
// TODO: declare forward + backward signatures once Tensor's design is set.
Tensor softmax_cross_entropy(const Tensor& a, const Tensor& b);

}  // namespace microtorch
