#include "microtorch/optim/adam.hpp"

namespace microtorch {

Adam::Adam(const std::vector<TensorPtr> &params, double lr,
           double beta1, double beta2, double eps)
    : params_(params), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0)
{
    for (const auto& p : params_) {
        m_.push_back(Eigen::MatrixXd::Zero(p->data.rows(), p->data.cols()));
        v_.push_back(Eigen::MatrixXd::Zero(p->data.rows(), p->data.cols()));
    }
}

void Adam::step() {
    t_ += 1; //sets the step counter to 1 since bias correction needs it to start at 1, not 0.

    for (size_t i = 0; i < params_.size(); ++i) {
        auto& p = params_[i];

        m_[i] = beta1_ * m_[i] + (1 - beta1_) * p->grad; //formula written in per-parameter form.
        v_[i] = beta2_ * v_[i] + (1 - beta2_) * p->grad.array().square().matrix(); //element-wise squaring of the gradient, then converted back to a matrix.

        double bias_correction1 = 1 - std::pow(beta1_, t_); //bias correction for the first moment
        double bias_correction2 = 1 - std::pow(beta2_, t_); //bias correction for the second moment

        Eigen::MatrixXd m_hat = m_[i] / bias_correction1; //bias-corrected first moment estimation
        Eigen::MatrixXd v_hat = v_[i] / bias_correction2; //bias-corrected second moment estimation

        p->data -= lr_ * (m_hat.array() / (v_hat.array().sqrt() + eps_)).matrix();
    }
}

void Adam::zero_grad() { //resets every parameter's gradient to zero so that no leak in gradients from previous back-passes occurs.
    for (auto& p : params_) {
        p->grad.setZero();
    }
}

}