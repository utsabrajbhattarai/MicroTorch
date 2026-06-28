#pragma once
// Tensor: wraps an Eigen matrix, carries data + grad + a record of how it
// was produced (for reverse-mode autodiff). This is the core object every
// op operates on.
//
// Not yet implemented -- this is a structural placeholder so the build
// graph is correct from day one. Fill in as you derive the design.

#include <Eigen/Dense>

namespace microtorch {

class Tensor {
public:
    // TODO: data, grad, and the computational-graph bookkeeping go here.
};

}  // namespace microtorch
