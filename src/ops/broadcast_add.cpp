#include "microtorch/ops/broadcast_add.hpp"
#include <cassert>

namespace microtorch {

// TODO: broadcast_add forward + backward implementation.

TensorPtr broadcast_add(const TensorPtr& a, const TensorPtr& b){


    //ensuring that the broadcast add is only MxN and 1xN and user isn't doing something silly atm
    assert(b->data.rows() == 1 && "broadcast_add: b must be a single row (1, N)");
    assert(a->data.cols() == b->data.cols() && "broadcast_add: a and b must have same #columns");

    //forward pass for output adding matrices
    //note that here we are adding B's single row to every row of A, this is to impersonate the only type of broadcast add we will be using for our models later on the addition fo further broadcasting rules to impersonate other typa matrix addition are independent features that can be further added but for the scope of our project we will be limiting to only [M,N]+{1,N] typa add used generally for GNN bias
    Eigen::MatrixXd out_matrix = a->data.rowwise() + b->data.row(0);

    //initializng a new tensor object to store the output+storing it in graph for prolonging the scope
    TensorPtr out = make_tensor(out_matrix);


    out -> parents = {a,b}; //defining the parents of the new output


    //using raw pointers here that point to same as shared_ptr so that there's no infinite loop of out storing out and a memory leak
    Tensor* ap = a.get();
    Tensor* bp = b.get();
    Tensor* outp = out.get();


    //storing thr backward_function of the ouput ie, how backward_function will be called 
    out->backward_fnc = [ap, bp, outp]() {
        ap->grad += outp->grad;                     //a's gradient are 1*previous_gradients (ie, out.grad)
        bp->grad += outp->grad.colwise().sum();     //b's gradient are sum of the columnwise 
    };

    return out;


}

}  // namespace microtorch
