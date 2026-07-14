#include "microtorch/Tensor.hpp"
#include "microtorch/gnn/gnn_model.hpp"

#include <cmath>
#include <random>



namespace microtorch {

    //helper function for the initialization of weights (He-init: used because suitable for ReLU)
    static Eigen::MatrixXd he_init(int rows, int cols){
        static std::mt19937 gen(95);    //mt19937 is just mercene twister random number generator static keyword preserves the state of the generator so in next function call a diff random number is called
        double stddev = std::sqrt(2.0 / rows);  //he:scale for relu sqqrt(2 / fan_in) fan_in->number of input to that layer
        std::normal_distribution <double> dist(0.0, stddev); //creating a normal distribution keeping mean 0 and standard deviation from he-scale

        Eigen::MatrixXd weights(rows, cols);    //creating the weight matrix to be returned
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                weights(i,j) = dist(gen);   //seeding using the normal distribution from the gen
            }
        }

        return weights;
    }


    //constructor:
    GNNModel::GNNModel(int in_features, int hidden, int num_classes){
        W1_ = make_tensor(he_init(in_features, hidden));    //weights for the first layer  ||  input-> features-features || output-> first GNN layer features
        W2_ = make_tensor(he_init(hidden, hidden));         //weights for the hidden layer ||  input-> first GNN layer features || output-> second GNN layer features
        W_out_ = make_tensor(he_init(hidden, num_classes)); //weights for the classifier layer||input->second GNN layer features|| output-> output classifier class

        b1_ = make_tensor(Eigen::MatrixXd::Zero(1, hidden)); //biases for the first layer
        b2_ = make_tensor(Eigen::MatrixXd::Zero(1, hidden));//biases for the second layer
        b_out_ = make_tensor(Eigen::MatrixXd::Zero(1, num_classes));//biases for the classifier layer
    }

    //parameters function(): for a vector which is directly passable to the optimizer or this function just vectorizes it
    std::vector<TensorPtr> GNNModel::parameters(){
        return { W1_, b1_,
                W2_, b2_,
                W_out_, b_out_
                };
    }

}  // namespace microtorch
