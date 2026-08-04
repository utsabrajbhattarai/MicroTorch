#include "microtorch/Tensor.hpp"
#include "microtorch/diffusion/diffusion_model.hpp"
#include "microtorch/ops.hpp"
#include "microtorch/data/toy_data.hpp"
#include "microtorch/optim/adam.hpp"
#include "microtorch/diffusion/noise_schedule.hpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <string>

using namespace microtorch;

const int N = 1000; //number of points
const int T = 200; //number of timestamps
const int hidden = 64;  //number of hidden neurons
const double lr = 0.0005;    //learning rate of adam optimizer

const int MAX_EPOCH = 250;

int main() {
    
    //loading/generating the toy data:
    Eigen::MatrixXd x0 = make_spiral(N); //noise is already a default arg

    //creating noise schedule
    NoiseSchedule ns = make_noise_schedule(T);

    //MODEL:
    DiffusionModel model(3, hidden, 2); //3 is input variables x,y,t 2 is output variables x_noise, y_noise

    //Optimizer:(using adam opt)
    Adam opt(model.parameters(), lr);

    //generator for timstep pick and forward noise
    std::mt19937 rng(95);

    //uniform distribution for noise t at each epoch
    std::uniform_int_distribution<int> t_dist(0, T - 1);

    //training loop:
    for (int epoch = 0; epoch <= MAX_EPOCH; epoch++){

        
        int t = t_dist(rng); //each epoch => random noise
        Eigen::MatrixXd noise_out; //training target
        Eigen::MatrixXd x_t = forward_noise(x0, t, ns, noise_out, rng); //filling noise_out and generating a noised x_t


        //making the input_tensor
        Eigen::MatrixXd input(N,3);
        input.leftCols(2) = x_t;    //filling inputs first 2 columns with noised x_t
        input.col(2).setConstant(t / static_cast<double>(T));   //normalizing t to make it range from 0 to 1
        TensorPtr input_tensor = make_tensor(input);

        TensorPtr pred = model.forward(input_tensor); //getting prediction using the forward layer

        TensorPtr loss = mse_loss(pred, noise_out); //comparing predicted with real noise using mse

        //BACKWARD + UPDATE WEIGHTS AND BIASES:
        opt.zero_grad(); //resetting gradient 0

        loss->backward(); //backward through all parameters

        opt.step(); //updating gradient using adam


        //print epoch::
        if (epoch % 10 == 0){
            std::cout << "epoch " << epoch << "  loss " << loss->data(0,0) << std::endl;
        }

    }


    

    return 0;
}
