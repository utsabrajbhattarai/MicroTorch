#pragma once
#include "microtorch/Tensor.hpp"
#include <cassert>

namespace microtorch {

//add: matrix addition of same shape matrix(basically an element wise addition)
TensorPtr add(const TensorPtr& a, const TensorPtr& b);

}  // namespace microtorch