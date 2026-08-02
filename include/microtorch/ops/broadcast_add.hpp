#pragma once
#include "microtorch/Tensor.hpp"
#include <cassert>

namespace microtorch {

//broadcast add: matrix addition usually for biases of a specific broadcasting rule
//note that parameters are actually TensorPtr& cause inside engine we will always be dealing with sharedptr so to address the graph and parents properly cause bare Tensor& is justa. stack copy of value and doesnt work with our backward() implementation
TensorPtr broadcast_add(const TensorPtr& a, const TensorPtr& b);

}  // namespace microtorch
