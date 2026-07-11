#include "microtorch/ops/neighbor_aggregation.hpp" 
#include <cassert>

namespace microtorch {
//here we are assuming that A is already normalized before being passed to this function
TensorPtr neighbor_aggregation(const Eigen::SparseMatrix<double>& A, const TensorPtr& X){

    //0) Assurance: for valid matrix multiplication
    assert(X->data.rows() == A.cols() && "MatMul: A's columns must equal X's row");


    //1) Forward pass
    //It's just a matrix multiplication of sparse matrix with our own matrix ie(N,N)x(N,F)=(N,F)
    //each node's row is combination of its neighbours feature row 
    Eigen::MatrixXd out_matrix = A * X->data;


    //2) Initializing the new object and sharedptr
    TensorPtr out = make_tensor(out_matrix);


    //3) Defining the parents of the new output
    out -> parents = {X}; //Here A represents relation of each node with every other on the graph which we dont need to backward though


    //4)Getting raw pointers so to prevent memory leak:
    Tensor* Xp = X.get();
    Tensor* outp = out.get();

    //5) Backward Pass
    //formula is exactly similar to one with matmul and gradient is calculated same way dL/dX must be shape (N,F) which works by (N,N)x(N,F)
    out->backward_fnc = [Xp, outp, A]() {  
        Xp->grad += A.transpose() * outp->grad;
    };
        

    return out;
}

}  // namespace microtorch
