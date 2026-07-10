#include "microtorch/ops/reduction.hpp"
#include <cassert>

namespace microtorch {
TensorPtr mean(const TensorPtr& a){

    //0) ensuring that the matrix isnt empty ecause mean divides by element count and smtg/0 is error worthy
    assert(a->data.size() > 0 && "reduction: cannot take the mean of an empty matrix");



    //1) Forward pass
    //the forward pass is just sum of each divided by size ie. the mean
    Eigen::MatrixXd out_matrix(1,1);
    out_matrix(0,0) = a->data.sum() / a->data.size(); 


    //2) Initializing the new object and sharedptr
    TensorPtr out = make_tensor(out_matrix);


    //3) Defining the parents of the new output
    out -> parents = {a};


    //4)Getting raw pointers so to prevent memory leak:
    Tensor* ap = a.get();
    Tensor* outp = out.get();


    //5) Backward Passs
    //since we have taken mean of whole into a single scalar so derivative for "a" is just broadcasting 1/size_of_matrix times the gradient of prev function

    out->backward_fnc = [ap, outp]() {                  
        double size = 1.0 / ap->data.size();   //for calcualting the derivate of the mean part
        ap->grad.array() += outp->grad(0,0) * size ;    //way to scalar broadcast to each cell 
    };

    return out;
}

}  // namespace microtorch
