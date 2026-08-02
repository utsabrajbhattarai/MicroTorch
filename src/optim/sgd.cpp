#include "microtorch/optim/sgd.hpp"

namespace microtorch {

    SGD :: SGD(const std::vector<TensorPtr> &params, double lr)
    {
        params_ = params;   //setting values from constructor to other variables
        lr_ = lr;
    }

    void SGD::step()
    {
        for(TensorPtr p : params_)
        {
            p->data  -= lr_ * p->grad; // changing weight a little (w = w0 - grad * lr)
        }
    }

    void SGD::zero_grad()
    {
        for(TensorPtr p : params_)
        {
            p->zero_grad();  //set gradient to zero for replacement with new value
        }
    }
}  