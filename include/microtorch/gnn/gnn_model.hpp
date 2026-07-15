#pragma once
#include "microtorch/Tensor.hpp"
#include <Eigen/SparseCore>
#include <vector>


namespace microtorch {

class GNNModel{
    private:
        TensorPtr W1_, b1_ ;    //Layer 1: first layer (X @ W1) + b1 ->relu 

        TensorPtr W2_, b2_;     //Layer 2: hidden layer (H @ W2) + b2 ->relu 

        TensorPtr W_out_, b_out_; //Classifier layer (H @ W_out) + b_out (no relu cause we need raw logits )

    public:
        //constructors parameters: in_features: nuber of input feature per node
        //hidden: hidden layer width
        //num_classes: output classes (number of categories in the output)
        GNNModel(int in_features, int hidden, int num_classes);


        //the full proof forward pass: consists of 2 GNN layers and a final classifier layer to generate logits
        //expected to return raw logits of shape (N, num_classes), softmax to be done seperately
        TensorPtr forward(const Eigen::SparseMatrix<double> & A, const TensorPtr& X);


        //to store the learnable tensors, for the optimizer to handle all the backwards of 
        std::vector<TensorPtr> parameters();
};


}   // namespace microtorch