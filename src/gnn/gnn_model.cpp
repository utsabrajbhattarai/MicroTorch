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

}  // namespace microtorch
