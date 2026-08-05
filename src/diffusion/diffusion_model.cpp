#include "microtorch/Tensor.hpp"
#include "microtorch/diffusion/diffusion_model.hpp"
#include "microtorch/ops.hpp"
#include <cmath>
#include <random>

namespace microtorch {

    Eigen::MatrixXd forward_noise(const Eigen::MatrixXd& x0, int t, const NoiseSchedule& ns, Eigen::MatrixXd& noise_out, std::mt19937& rng){

        //Step 1: fill noise_out with a normal noise (same shape as x0)
        noise_out = Eigen::MatrixXd(x0.rows(), x0.cols());
        std::normal_distribution<double> dist(0.0, 1.0);

        for(int i = 0; i < noise_out.rows(); i++){
            for(int j = 0; j < noise_out.cols(); j++){
                noise_out(i,j) = dist(rng);
            }
        }

        //Step 2: calculating coefficients of equation:
        double c1 = sqrt(ns.alpha_bar[t]);
        double c2 = sqrt(1.0 - ns.alpha_bar[t]);

        //Step 3: blending (creating x_t and return)
        Eigen::MatrixXd x_t = c1 * x0 + c2 * noise_out;

        return x_t;

    }

    //he-init for the weights:(using ReLU)
    static Eigen::MatrixXd he_init(int rows, int cols) {
        static std::mt19937 gen(95);
        double stddev = std::sqrt(2.0 / rows);  
        std::normal_distribution<double> dist(0.0, stddev);
        Eigen::MatrixXd w(rows, cols);  //dummy weight
        for (int i = 0; i < rows; ++i){
            for (int j = 0; j < cols; ++j){
                w(i, j) = dist(gen);
            }
        }
        return w;
    }

    //constructor to initialize weights and biases
    DiffusionModel::DiffusionModel(int in_features, int hidden, int out_features){
        W1_ = make_tensor(he_init(in_features, hidden));
        b1_ = make_tensor(Eigen::MatrixXd::Zero(1, hidden));
        W2_ = make_tensor(he_init(hidden, hidden));
        b2_ = make_tensor(Eigen::MatrixXd::Zero(1, hidden));
        W_out_ = make_tensor(he_init(hidden, out_features));
        b_out_ = make_tensor(Eigen::MatrixXd::Zero(1, out_features));

    }

    //returning the parameters list 
    std::vector<TensorPtr> DiffusionModel::parameters() {
    return { W1_, b1_, W2_, b2_, W_out_, b_out_ };
    }
    //the forward layer:
    TensorPtr DiffusionModel::forward(const TensorPtr& input) {
        //Layer 1:
        TensorPtr z1 = matmul(input, W1_); //
        TensorPtr pre_layer1 = broadcast_add(z1, b1_);
        TensorPtr layer1 = relu(pre_layer1);

        //Layer 2:
        TensorPtr z2 = matmul(layer1, W2_);
        TensorPtr pre_layer2 = broadcast_add(z2, b2_);
        TensorPtr layer2 = relu(pre_layer2);

        //Layer3:
        TensorPtr z3 = matmul(layer2, W_out_);
        TensorPtr final_layer = broadcast_add(z3, b_out_);

        return final_layer;

    }


    //sampling logic:
    Eigen::MatrixXd sample(DiffusionModel& model, const NoiseSchedule& ns, int n, int T, std::mt19937& rng){

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
    }

}
  // namespace microtorch
