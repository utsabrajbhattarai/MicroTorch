#include "microtorch/ops/softmax_cross_entropy.hpp"
#include <cassert>

namespace microtorch {

    //overloaded to handle masked and unmasked case(masked case = having unlabelled data, unmasked case = all data labelled)
TensorPtr softmax_cross_entropy(const TensorPtr& logits, const Eigen::MatrixXd& targets, const Eigen::MatrixXd& mask, double num_labeled){

    //0) Assurance: logits and one-hot targets must match for elementwise operations
    assert(logits->data.rows()==targets.rows() && logits->data.cols()==targets.cols()
           && "softmax_cross_entropy: logits and targets must have the same shape");

    //1) Forward pass
    //the operations are SOFTMAX()->CROSS_ENTROPY() (softmax being (e^z_i)/(sum_k(e^z_k))) and cross_entropy being (-sum_k(y_k*log_k))
    //firstly we shift the logit to max being 0 so to prevent e^max swinging towards infinity and breaking our system

    //softmax:
    Eigen::MatrixXd shifted_logits = logits->data.array().colwise() - logits->data.rowwise().maxCoeff().array(); //.colwise broadcasting each column subtraction .rowise dimensionally reducing from (N,C) to (N,1) like a column vector
    Eigen::MatrixXd exp = shifted_logits.array().exp(); //element wise exponentiation
    Eigen::MatrixXd probs = exp.array().colwise() / exp.rowwise().sum().array();    //similar broadcasting and reducing like shifted logits but different operations

    //cross_entropy:
    double loss = -(targets.array() * probs.array().log()).sum() / num_labeled; //should only be divided by the no. of rows that have an impact on loss and are not (0,0)

    Eigen::MatrixXd out_matrix(1,1);
    out_matrix(0,0) = loss;


    //2) Initializing the new object and sharedptr
    TensorPtr out = make_tensor(out_matrix);


    //3) Defining the parents of the new output
    out -> parents = {logits}; //we only need to backward though logits and the one hot encoding of true matrix or targets


    //4)Getting raw pointers so to prevent memory leak:
    Tensor* logitsp = logits.get();
    Tensor* outp = out.get();

    //5) Backward Pass
    //the reason why we combined softmax and cross entropy was for sake of this backward
    //due to clean mathematics and derivaitve there is no jacobian term but just P-Y for each node and (P-Y)/N for a row
    //where P=probs matrx (N,C) and Y is targets (N,C) for the sake of this backward we also need to capture the value of probs 
    //mask is to ensure the unlabelled rows grads are zeroed out ie, unlabelled grads arent updated (its 0 for unknown and 1 for labelled)
    out->backward_fnc = [logitsp, outp, targets, probs, mask, num_labeled]() {  
        double scale = outp->grad(0,0) / num_labeled;
        Eigen::MatrixXd grad = (probs - targets).array().colwise() * mask.col(0).array();
        logitsp->grad += scale * grad;
    };
        
    return out;

}



TensorPtr softmax_cross_entropy(const TensorPtr& logits, const Eigen::MatrixXd& targets) {
    Eigen::MatrixXd mask = Eigen::MatrixXd::Ones(targets.rows(), 1);

    return softmax_cross_entropy(logits, targets, mask, targets.rows()); // mask being all 1's means each row contributes and if in forward N=num_labelled then it will work like this overloaded logic
}



}  // namespace microtorch
