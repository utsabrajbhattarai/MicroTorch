#pragma once
#include "microtorch/Tensor.hpp"
#include <functional>
#include <cmath>
#include <algorithm>

namespace microtorch {

    //To check if the gradients are not silently breaking down by comparing analytic gradient with the numerical approx.
    //returns max difference between the gradients mentioned
    double gradient_check(std::function<TensorPtr(TensorPtr)> loss_fn,  //Wrapper that is basically lamda replication of whole forward() for any list of operations for eg (Tensorptr x) {return func(list of args);}
                        const TensorPtr& input,                         //input tensors whose gradient we need to check
                        double eps =1e-6);                              //approx for derivatives                       

}