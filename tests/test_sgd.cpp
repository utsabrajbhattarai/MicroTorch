#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"
#include "microtorch/optim/sgd.hpp"
#include <catch2/catch_approx.hpp>

TEST_CASE("SGD converges on toy quadratic f(x) = x^2", "[sgd]")
{
    Eigen::MatrixXd initial_value(1, 1); //make matrix 1*1 to store one value 5.0 coz tensor is always matrix
    initial_value(0, 0) = 5.0;
    microtorch::TensorPtr x = microtorch::make_tensor(initial_value);
   

    microtorch::SGD optimizer({x}, 0.1);   // one parameter, learning rate 0.1

    for (int i = 0; i < 100; i++)
    {
        x->grad = 2.0 * x->data;   // self calculated gardient of x^2 = 2x for test case
        optimizer.step();
    }

    REQUIRE(x->data(0, 0) == Catch::Approx(0.0).margin(1e-3));
}