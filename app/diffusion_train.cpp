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
const double lr = 0.001;    //learning rate of adam optimizer



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


    return 0;
}
