#pragma once
#include "microtorch/Tensor.hpp"
#include <vector>

namespace microtorch {

// SGD -- dead simple, no per-parameter state needed beyond the gradient
// already stored on each Tensor.
// TODO: define once Tensor's parameter-list convention is settled.
class SGD {
public:
    // TODO: constructor (params, learning_rate), step(), zero_grad().
};

}  // namespace microtorch
