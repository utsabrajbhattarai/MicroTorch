#include "microtorch/Tensor.hpp"
#include "microtorch/diffusion/diffusion_model.hpp"
#include "microtorch/ops.hpp"
#include "microtorch/data/toy_data.hpp"
#include "microtorch/optim/adam.hpp"
#include "microtorch/diffusion/noise_schedule.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>
#include <string>

using namespace microtorch;

const int N = 1000; //number of points
const int T = 1000; //number of timestamps
const int hidden = 256;  //number of hidden neurons
const double lr = 0.005;    //learning rate of adam optimizer

const int MAX_EPOCH = 2000;
const int TIMESTEPS_PER_EPOCH = 16;   //random t's per step
const int K = 4;   //number of sin/cos frequency pairs => time embedding is 2*K wide, so input width is 2 + 2*K = 10

//loading/generating the toy data:
Eigen::MatrixXd x0 = make_light_spiral(N); //noise is already a default arg
const std::string SHAPE_NAME = "simple_spiral";   //per run changing this to a certain shape to directly save shape in frames_shape path

//sampling loop for generating sample from predicted noise and generated a frame by frame points
std::vector<Eigen::MatrixXd> generate(DiffusionModel& model, const NoiseSchedule& ns, int n, int T, std::mt19937& rng);


//this actually turns one scalar timestep t into 2*k sin/cos waves so the net can tell noise levels apart (like transformer positional encoding)
Eigen::VectorXd time_embedding(int t, int k, int T){
    double t_norm = t / static_cast<double>(T);   //squishing t into 0..1 first
    Eigen::VectorXd emb(2 * k);   //2 features per frequency (one sin, one cos)
    for (int i = 0; i < k; i++){
        double freq = std::pow(2, i) * M_PI;   //frequencies double each step so every pair looks at t on a different scale
        emb(2 * i)     = std::sin(t_norm * freq);
        emb(2 * i + 1) = std::cos(t_norm * freq);
    }
    return emb;
}

//this glues x and the time embedding into one (N, 2+2k) block. same func called by BOTH train and sample so the encoding can never drift apart
Eigen::MatrixXd build_input(const Eigen::MatrixXd& x, int t, int k, int T){
    Eigen::VectorXd emb = time_embedding(t, k, T);
    Eigen::MatrixXd input(x.rows(), 2 + 2 * k);
    input.leftCols(2) = x;   //first two cols stay the actual points
    input.rightCols(2 * k) = emb.transpose().replicate(x.rows(), 1);   //same embedding copied down onto every row
    return input;
}

int main() {

    //creating noise schedule
    NoiseSchedule ns = make_noise_schedule(T);

    //MODEL:
    DiffusionModel model(2 + 2 * K, hidden, 2); //input is x,y + 2*K time-embedding cols, 2 is output variables x_noise, y_noise

    //Optimizer:(using adam opt)
    Adam opt(model.parameters(), lr);

    //generator for timstep pick and forward noise
    std::mt19937 rng(120);

    //uniform distribution for noise t at each epoch
    std::uniform_int_distribution<int> t_dist(0, T - 1);


    //training loop:
    for (int epoch = 0; epoch <= MAX_EPOCH; epoch++){

        TensorPtr total_loss = nullptr; //total loss per epoch(across k t's)

        // inner loop: accumulate loss over several random timesteps
        for (int k = 0; k < TIMESTEPS_PER_EPOCH; k++) {
            int t = t_dist(rng); //each epoch => random noise
            Eigen::MatrixXd noise_out; //training target
            Eigen::MatrixXd x_t = forward_noise(x0, t, ns, noise_out, rng); //filling noise_out and generating a noised x_t


            //making the input_tensor
            Eigen::MatrixXd input = build_input(x_t, t, K, T); //x_t in first 2 cols, sinusoidal time embedding in the rest
            TensorPtr input_tensor = make_tensor(input);

            TensorPtr pred = model.forward(input_tensor); //getting prediction using the forward layer

            TensorPtr loss = mse_loss(pred, noise_out); //comparing predicted with real noise using mse

            //accumulate: first iteration seeds it, rest add on
            total_loss = (total_loss == nullptr) ? loss : add(total_loss, loss);

        }
        //BACKWARD + UPDATE WEIGHTS AND BIASES:
        opt.zero_grad(); //resetting gradient 0

        total_loss->backward(); //backward through all parameters

        opt.step(); //updating gradient using adam


        //print epoch::
        if (epoch % 10 == 0){
            std::cout << "epoch " << epoch << "  avg_loss " << total_loss->data(0,0) / TIMESTEPS_PER_EPOCH << std::endl;
        }

    }

    //sampling function:
    std::vector<Eigen::MatrixXd> frames = generate(model, ns, N, T, rng);

    //dump every frame: columns are frame_index,x,y  (one row per point per frame)
    std::filesystem::create_directories("frames");   //doesnt recreate if already created a c++-17 typa code
    std::ofstream out("frames/frames_" + SHAPE_NAME + ".csv");  //path for final file
    for (size_t f = 0; f < frames.size(); f++){
        for (int i = 0; i < N; i++){
            out << f << "," << frames[f](i, 0) << "," << frames[f](i, 1) << "\n";
        }
    }
    out.close();

    return 0;
}




//sampling logic:
std::vector<Eigen::MatrixXd> generate(DiffusionModel& model, const NoiseSchedule& ns, int n, int T, std::mt19937& rng){

    std::vector<Eigen::MatrixXd> frames;   //basically a (n,2) snapshot per timestep

    //the final x to return
    Eigen::MatrixXd x(n,2);

    std::normal_distribution<double> dist(0.0, 1.0); //normal distribution
    for (int i = 0; i < n; i++ ){
        for (int j = 0; j < 2; j++ ){ 
            x(i,j) = dist(rng); //each element holds a random noise at first
        }
    }

    frames.push_back(x);   //frame 0 = pure noise

    for (int t = T - 1; t >= 0; t--) {

        //same like forward creating the input Tensor
        Eigen::MatrixXd input = build_input(x, t, K, T); //exact same encoding as training so train/sample never drift apart
        TensorPtr input_tensor = make_tensor(input);


        TensorPtr pred = model.forward(input_tensor); //prediction

        //Reversing the noise: //mathematical formulas:
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

        if (t % 4 == 0 || t == 0) { //only save every 4th step to reduce the file size and processing speed enhancement also 0th or the last step too 
            frames.push_back(x); //snapshot this timestep also
        }

    }
    return frames;   //return all the frames, not just final x
}