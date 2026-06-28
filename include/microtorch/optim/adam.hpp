#pragma once
#include "microtorch/Tensor.hpp"
#include <vector>

namespace microtorch {

// Adam -- needs per-parameter moving averages (m, v) stored across steps,
// more state than SGD. Build SGD first; this is the week 2 add-on.
// TODO: define once Tensor's parameter-list convention is settled.
class Adam {
public:
    // TODO: constructor (params, lr, beta1, beta2, eps), step(), zero_grad().
};

}  // namespace microtorch
