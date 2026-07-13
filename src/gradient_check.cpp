#include "microtorch/gradient_check.hpp"

namespace microtorch{

    double gradient_check(std::function<TensorPtr(TensorPtr)> loss_fn, const TensorPtr& input, double eps){

        //Zeroing the input gradient just in case user passed with non-zeroed values:
        input->zero_grad();

        //Computing the analytic gradient
        TensorPtr loss = loss_fn(input);
        loss->backward();
        Eigen::MatrixXd analytic_gradient = input->grad;    //copying the input gradients for further comparision later on

        //Computing numeric gradient using central finite differences
        //using formula: f'(x) = {f(x+h)-f(x-h)}/2h

        double max_error = 0.00;
        for(int i = 0; i < input->data.rows(); i++){

            for(int j = 0; j < input->data.cols(); j++){

                double x = input->data(i,j);

                //calculating f(x+h)
                input->data(i,j) = x + eps;
                double loss_plus = loss_fn(input)->data(0,0);

                //calculating f(x-h)
                input->data(i,j) = x - eps;
                double loss_minus = loss_fn(input)->data(0,0);

                //restore original value inside the data 
                input->data(i,j) = x;

                //calculate the numerical derivative:
                double numerical_gradient = (loss_plus - loss_minus) / (2 * eps);

                //get the maximum error:
                max_error = std::max(max_error, std::abs(numerical_gradient - analytic_gradient(i,j)));

            }
        }

        return max_error;

    }

}