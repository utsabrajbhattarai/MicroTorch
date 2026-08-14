#include "microtorch/optim/adam.hpp"
#include <cmath>

namespace microtorch {

    //constructor: pass params to the base, init hyperparams, allocate moment
    //matrices matching each parameter's shape (all zeros to start)
    Adam::Adam(const std::vector<TensorPtr>& params, double lr,
               double beta1, double beta2, double eps)
        : Optimizer(params), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0) {
        for (auto& p : params_) {
            m_.push_back(Eigen::MatrixXd::Zero(p->data.rows(), p->data.cols()));
            v_.push_back(Eigen::MatrixXd::Zero(p->data.rows(), p->data.cols()));
        }
    }

    //Adam update: exponential moving averages of gradient (m) and squared
    //gradient (v), bias-corrected, then the parameter step
    void Adam::step() {
        t_++;   //advance timestep for bias correction
        for (size_t i = 0; i < params_.size(); i++) {
            Eigen::MatrixXd g = params_[i]->grad;

            //update biased first and second moment estimates
            m_[i] = beta1_ * m_[i] + (1.0 - beta1_) * g;
            v_[i] = beta2_ * v_[i] + (1.0 - beta2_) * g.array().square().matrix();

            //bias correction
            Eigen::MatrixXd m_hat = m_[i] / (1.0 - std::pow(beta1_, t_));
            Eigen::MatrixXd v_hat = v_[i] / (1.0 - std::pow(beta2_, t_));

            //parameter update
            params_[i]->data.array() -= lr_ * m_hat.array() / (v_hat.array().sqrt() + eps_);
        }
    }

}  // namespace microtorch