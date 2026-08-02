#pragma once

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <vector>
#include <unordered_set>

namespace microtorch {

class Tensor;  //forward declaration so that alias below can use it
using TensorPtr = std::shared_ptr<Tensor>;  //easy to use Tensor Ptr shortcut btw this extennds lifetime of a certain variable

class Tensor {
public:
    Eigen::MatrixXd data;  //actual values of the tensor
    Eigen::MatrixXd grad;  //gradient of the tensor -same shape as the data

    //autograd bookkeeps of previous ops
    std::vector<TensorPtr> parents;  //lists the operation this tensor was actually computed from
    std::function<void()> backward_fnc;  //basically a backward function to push the gradients to the parents tensors


    //Eigen::MatrixXd means X rows,columns ie. X means dynamic (size decided at runtime) and d means double ie. the datatype
    explicit Tensor(const Eigen::MatrixXd& values) //tensor constructor that takes in matrix of values to initialize the tensor.data and tensor.grad
        : data(values),
        grad(Eigen::MatrixXd::Zero(values.rows(), values.cols())),  //this initializes grad as same size of data  but all zeroes 
        backward_fnc(nullptr) {}    //every backward func initialized to not have a backward

    void zero_grad(); //sets the gradient to zero so that no leak in gradients from prev. back-passes 
    void backward();    //the backward operations including 

    

};
//makes a heap Tensor and hands back a shared_ptr to it
inline TensorPtr make_tensor(const Eigen::MatrixXd& values) {   //need to keep it outside tensor class so to create a require tensor
    return std::make_shared<Tensor>(values);
}

}  // namespace microtorch
