#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "microtorch/Tensor.hpp"
#include "microtorch/optim/adam.hpp"

using namespace microtorch;

TEST_CASE("Adam optimizer forward is correct", "[adam]") { //minimize f(x) = x^2, gradient = 2x, starting at x=5
    Eigen::MatrixXd start(1, 1); //A 1x1 matrix to hold the starting value of x; x = 5.0
    start(0, 0) = 5.0; //starting at x=5.0, so the gradient is 10.0, and the next step should be x = 5 - 0.1 * 10 = 4.0, and so on.
    TensorPtr x = make_tensor(start);

    Adam opt({x}, 0.1); //construct an Adam optimizer with the tensor x and a learning rate of 0.1

    for (int i = 0; i < 200; ++i) {
        x->grad(0, 0) = 2 * x->data(0, 0);  //set gradient to 2x, which is the derivative of x^2
        opt.step(); //applies Adam update using the gradient from above, and updates x->data to the new value of x.
    }

    REQUIRE(x->data(0, 0) == Catch::Approx(0.0).margin(1e-3)); //After 200 steps, x should be very close to 0.0, since we are minimizing x^2. The margin is almost zero, but not exactly zero, to account for numerical precision errors.
}
