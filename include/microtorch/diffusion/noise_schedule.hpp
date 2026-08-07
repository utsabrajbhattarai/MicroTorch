#pragma once
#include <Eigen/Dense>


namespace microtorch {

    struct NoiseSchedule{
        int T;                  //Total Timestep 
        Eigen::VectorXd beta;       //beta=> noise added at each step t
        Eigen::VectorXd alpha;      //alpha->(1-beta)=>signal left at t
        Eigen::VectorXd alpha_bar;  //cumulative product of alpha
    };

    NoiseSchedule make_noise_schedule(int T, double beta_start = 1e-4, double beta_end=0.02);

} //microtorch