#include "microtorch/Tensor.hpp"
#include "microtorch/diffusion/diffusion_model.hpp"
#include "microtorch/ops.hpp"
#include "microtorch/data/toy_data.hpp"
#include "microtorch/optim/adam.hpp"
#include "microtorch/diffusion/noise_schedule.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
#include <string>

using namespace microtorch;

const int N = 1000; //number of points
const int T = 200; //number of timestamps
const int hidden = 64;  //number of hidden neurons
const double lr = 0.0005;    //learning rate of adam optimizer

const int MAX_EPOCH = 1000;


//sampling loop for generating sample from predicted noise 
Eigen::MatrixXd generate(DiffusionModel& model, const NoiseSchedule& ns, int n, int T, std::mt19937& rng);

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

    //sampling function:
    Eigen::MatrixXd generated = generate(model, ns, N, T, rng);
    std::ofstream out("generated.csv"); //saving points to a csv file
    for (int i = 0; i < N; i++){
        out << generated(i, 0) << "," << generated(i, 1) << "\n";
    }
    out.close();



    return 0;
}




//sampling logic:
Eigen::MatrixXd generate(DiffusionModel& model, const NoiseSchedule& ns, int n, int T, std::mt19937& rng){

    //the final x to return
    Eigen::MatrixXd x(n,2);

    std::normal_distribution<double> dist(0.0, 1.0); //normal distribution
    for (int i = 0; i < n; i++ ){
        for (int j = 0; j < 2; j++ ){ 
            x(i,j) = dist(rng); //each element holds a random noise at first
        }
    }

    for (int t = T - 1; t >= 0; t--) {

        //same like forward creating the input Tensor
        Eigen::MatrixXd input(n,3);
        input.leftCols(2) = x;    //filling inputs first 2 columns with noisy x
        input.col(2).setConstant(t / static_cast<double>(T));   //normalizing t to make it range from 0 to 1
        TensorPtr input_tensor = make_tensor(input);


        TensorPtr pred = model.forward(input_tensor); //prediction

        //Reversing the noise:
        double alpha_t = ns.alpha[t];
        double alpha_bar_t = ns.alpha_bar[t];
        double beta_t = ns.beta[t];

        //denoisig update: removing a scaled portion of predicted noise
        Eigen::MatrixXd pred_noise = pred->data;
        Eigen::MatrixXd mean = (1.0 / std::sqrt(alpha_t)) * (x - (beta_t / std::sqrt(1.0 - alpha_bar_t)) * pred_noise);

        //fresh randomness: to make it explore distribution better
        if (t > 0){
            Eigen::MatrixXd z(n,2);
            for (int i = 0; i < n; i++ ){
                for (int j = 0; j < 2; j++ ){ 
                    z(i,j) = dist(rng); //each element holds a random noise at first
                }
            }
            double sigma = std::sqrt(beta_t); //noise is added based on how far off is t from 0

            x = mean + sigma * z;
        }
        else {
            x = mean; //no noise added in last step;
        }


    }
    return x;
}