#include "microtorch/ops/relu.hpp"
#include <cassert>

namespace microtorch {

TensorPtr relu(const TensorPtr& a){
    
    //0) Assurance of not being a null pointer:
    assert(a && "relu: input tensor is null");

    //1) forward pass
    //since we have to check elementwise of the a layer then we have to use .array
    //max of 0 or a does the relu logic for negative becoming 0 and positive remaining same | the last .matrix is to convert array back to matrix
    Eigen::MatrixXd out_matrix = a->data.array().max(0.0).matrix();

    //2) Initializing the new object and sharedptr
    TensorPtr out = make_tensor(out_matrix);


    //3) Defining the parents of the new output
    out -> parents = {a};


    //4)Getting raw pointers so to prevent memory leak:
    Tensor* ap = a.get();
    Tensor* outp = out.get();


    //5) Backward function:
    //the function is x=x for (+ve) and x = 0 for (-ve) so derivative is 1 for +ve number and 0 for -ve, for sake of convention we'll give 0 for derivative of 0

    out->backward_fnc=[ap,outp](){
        Eigen::MatrixXd mask = (ap->data.array()>0.0).cast<double>();   // mask is array of boolean value converted to double type which satisfied 0 or 1 typa condtiiton
        ap->grad  += (mask.array() * outp->grad.array()).matrix();           //adding in form of array cause matrix addition may break this and we just want a elementwise addition
    };
    return out;
};

}  // namespace microtorch
