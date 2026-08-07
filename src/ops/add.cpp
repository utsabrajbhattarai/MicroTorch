#include "microtorch/ops/add.hpp"
#include <cassert>

namespace microtorch {

    TensorPtr add(const TensorPtr& a, const TensorPtr& b){
        //asserting that matrix are of same shape due to same-shape element-wise addition
        assert(a->data.rows() == b->data.rows() && a->data.cols() == b->data.cols()
            && "add: a and b must have the same shape");

        Eigen::MatrixXd out_matrix = a->data + b->data;   //plain matrix addition
        TensorPtr out = make_tensor(out_matrix);


        out->parents = {a, b};


        Tensor* ap = a.get();
        Tensor* bp = b.get();
        Tensor* outp = out.get();

        
        out->backward_fnc = [ap, bp, outp]() {
            ap->grad += outp->grad;   //gradient of a+b w.r.t. a is 1
            bp->grad += outp->grad;   //and w.r.t. b is also 1
        };
        return out;
    }


}
