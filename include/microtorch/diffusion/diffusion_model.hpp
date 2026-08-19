#pragma once
#include "microtorch/Tensor.hpp"
#include "microtorch/diffusion/noise_schedule.hpp"
#include <vector>
#include "microtorch/ops.hpp"
#include <random>


namespace microtorch {

    //forward of Diffsion=> Returns noisy x_t ie, data for training and also fills noise_out with noise
    Eigen::MatrixXd forward_noise(const Eigen::MatrixXd& x0, int t, const NoiseSchedule& ns, Eigen::MatrixXd& noise_out, std::mt19937& rng);


    //Diffusion model that handles the MLP portion of predicting noise at any step t
    class DiffusionModel {

    private:
        TensorPtr W1_, b1_;//Weights between input and first hidden layer
        TensorPtr W2_, b2_;//between first and second hidden layer
        TensorPtr W_out_, b_out_;//between second hidden layer and output

    public:
        //in_features = input width per point = 10  (x, y, normalized-t)
        //hidden = no. hidden layer neurons (eg 64)
        //out_features =  output width each point = 2  (predicted x-noise, y-noise)
        DiffusionModel(int in_features, int hidden, int out_features);

        //predicts the noise for each input row not that input is (n, 10), output (n, 2).
        //just a simple and plain MLP.
        TensorPtr forward(const TensorPtr& input);

        std::vector<TensorPtr> parameters(); //parameter list

        
    
    };


}