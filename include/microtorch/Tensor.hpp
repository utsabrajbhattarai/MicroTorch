#pragma once

#include <Eigen/Dense>

namespace microtorch {

class Tensor {
public:
    Eigen::MatrixXd data;  //actual values of the tensor
    Eigen::MatrixXd grad;  //gradient of the tensor -same shape as the data

    //Eigen::MatrixXd means X rows,columns ie. X means dynamic (size decided at runtime) and d means double ie. the datatype
    explicit Tensor(const Eigen::MatrixXd& values) //tensor constructor that takes in matrix of values to initialize the tensor.data and tensor.grad
        : data(values),
        grad(Eigen::MatrixXd::Zero(values.rows(), values.cols())) {}  //this initializes grad as same size of data  but all zeroes 

    void zero_grad(); //sets the gradient to zero so that no leak in gradients from prev. back-passes 
};

}  // namespace microtorch
