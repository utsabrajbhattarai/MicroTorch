#pragma once
#include "microtorch/Tensor.hpp"
#include <Eigen/SparseCore>

namespace microtorch {

// neighbor aggregation : sparse adjacency matmul specific matmul for node relationship in GNN
TensorPtr neighbor_aggregation(const Eigen::SparseMatrix<double>& A, const TensorPtr& X);

}  // namespace microtorch
