#include "microtorch/ops/mse_loss.hpp"
#include <cassert>

namespace microtorch {

TensorPtr mse_loss(const TensorPtr& pred, const Eigen::MatrixXd& target){

    //0) pred and target must line up for the subtraction to make sense
    assert(pred->data.rows()==target.rows() && pred->data.cols()==target.cols()
           && "mse_loss: pred and target must have the same shape");
    


    //1) Forward pass
    //loss = sum((y_pred - y_true)^2) /N also called Mean Squared Error Loss
    //squaredNorm allows to get sum of squares of all elements basically the easy way of numerator and data.size is just the N for mean
    Eigen::MatrixXd out_matrix(1,1);    //it must be a scalar or 1,1
    out_matrix(0,0)  = (pred->data - target).squaredNorm()/pred->data.size();


    //2) Initializing the new object and sharedptr
    TensorPtr out = make_tensor(out_matrix);


    //3) Defining the parents of the new output

    out -> parents = {pred}; //note that unlike other ops target is not in parent because it's a fixed constant and we dont really need it in the graph or find its gradient


    //4)Getting raw pointers so to prevent memory leak:
    Tensor* predp = pred.get();
    Tensor* outp = out.get();

    //5) Backward Pass
    //dL/dy_pred = 1/N * 2 * (y_pred - y_true) 

    //target is a normal pointer(not Shared) it dies when mse_loss return something but the backward lambda runs later
    //so the lambda must capture target by copy and not by the pointer value otherwise we will be pointing at a dead matrix
    out->backward_fnc = [predp, outp, target]() {  
        double scalar = (2.0/predp->data.size());
        predp->grad += (outp->grad(0,0) * scalar) * (predp->data - target);
    };
        

    return out;
}

}  // namespace microtorch
