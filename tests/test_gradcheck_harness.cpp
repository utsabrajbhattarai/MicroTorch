#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"
#include "microtorch/gradient_check.hpp"
#include "microtorch/ops/relu.hpp"
#include "microtorch/ops/reduction.hpp"   // for mean

using namespace microtorch;

//we will be implementing a broken relu just for sanity check that our gradient harness does flag incorrect gradients:
//our broken relu will be that in negative region (it forgets the >0 mask). This exists purely so the harness has a kown bad op to catch

static TensorPtr relu_broken(const TensorPtr& a) {
    Eigen::MatrixXd o = a->data.array().max(0.0).matrix();
    TensorPtr out = make_tensor(o);
    out->parents = { a };
    Tensor* ap = a.get();
    Tensor* op = out.get();
    out->backward_fnc = [ap, op]() {
        ap->grad += op->grad;   //on purpose bug here no conditions for 0 
    };
    return out;
}

TEST_CASE("gradient_check agrees with a correct backward", "[gradient_check]") {
    Eigen::MatrixXd X(3,4);
    X << -2, 1, -0.5, 3,   0.2, -1, 4, -0.1,   5, -3, 0.7, 2;
    auto x = make_tensor(X);
    double err = gradient_check([](TensorPtr t){ return mean(relu(t)); }, x);
    REQUIRE(err < 1e-6);
}

TEST_CASE("gradient_check catches a deliberately wrong backward", "[gradient_check]") {
    Eigen::MatrixXd X(3,4);
    X << -2, 1, -0.5, 3,   0.2, -1, 4, -0.1,   5, -3, 0.7, 2;
    auto x = make_tensor(X);
    double err = gradient_check([](TensorPtr t){ return mean(relu_broken(t)); }, x);
    REQUIRE(err > 1e-3);
}