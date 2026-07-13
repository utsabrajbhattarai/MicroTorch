#include "microtorch/gradient_check.hpp"

namespace microtorch{

    double gradient_check(std::function<TensorPtr(TensorPtr)> loss_fn, const TensorPtr& input, double eps = 1e-6){

        //Zeroing the input gradient just in case user passed with non-zeroed values:
        input->zero_grad();

        //Computing the analytic gradient
        TensorPtr loss = loss_fn(input);
        loss->backward();
        Eigen::MatrixXd analytic_gradient = input->grad;    //copying the input gradients for further comparision later on


    }

}