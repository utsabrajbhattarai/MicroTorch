#include "microtorch/ops/matmul.hpp"
#include <cassert>

namespace microtorch {

TensorPtr matmul(const TensorPtr& a, const TensorPtr& b){

    //0) ensuring that the matrix multiplication is valid
    assert(b->data.rows() == a->data.cols() && "MatMul: A's columns must equal B's row");


    //1) Forward pass
    //eigen has overloaded * operator to allow matrix multiplication so using that in our forward pass
    Eigen::MatrixXd out_matrix = a->data * b->data;


    //2) Initializing the new object and sharedptr
    TensorPtr out = make_tensor(out_matrix);


    //3) Defining the parents of the new output
    out -> parents = {a,b};


    //4)Getting raw pointers so to prevent memory leak:
    Tensor* ap = a.get();
    Tensor* bp = b.get();
    Tensor* outp = out.get();


    //5) Backward Passs
    //The derivative of certain matrix multiplication with respect to a matrix is:
    //  M = AxB
    //  Let, G = dL/dM ie. gradient of M with respect to loss function
    //  dM/dA = G x B'  where x represents matmul and ' represents transpose 
    //  dM/dB = A' x MG rest same as above

    out->backward_fnc = [ap, bp, outp]() {                  //lets say A shape is [M,k] , B is [k, N] G is [M,N]
        ap->grad += outp->grad * (bp->data).transpose();    //A.grad shape becomes [M,N]*[n,k] = [M,k]
        bp->grad += (ap->data).transpose() * outp->grad;    //B.grad shape becomes [k,M]*[M,N] = [k,N]
    };

    return out;
}

}  // namespace microtorch
