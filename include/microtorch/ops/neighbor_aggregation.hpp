#pragma once
#include "microtorch/Tensor.hpp"

namespace microtorch {

// neighbor aggregation -- sparse adjacency matmul, the one GNN-specific op
// TODO: declare forward + backward signatures once Tensor's design is set.
Tensor neighbor_aggregation(const Tensor& a, const Tensor& b);

}  // namespace microtorch
