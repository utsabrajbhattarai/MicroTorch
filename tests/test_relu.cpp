#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"
#include "microtorch/ops/relu.hpp"

using namespace microtorch;

TEST_CASE("relu: forward clamps negatives and backward masks gradient", "[relu]") {
    // deliberately mix positives, negatives, and a zero
    Eigen::MatrixXd A(2,3);
    A << -2.0,  1.0, -0.5,
          3.0,  0.0,  4.0;

    auto a   = make_tensor(A);
    auto out = relu(a);

    SECTION("forward: negatives and zero -> 0, positives unchanged") {
        Eigen::MatrixXd expected(2,3);
        expected << 0.0, 1.0, 0.0,
                    3.0, 0.0, 4.0;
        REQUIRE(out->data.isApprox(expected));
    }

    SECTION("backward: gradient passes where input>0, dies where input<=0") {
        out->backward();   // seeds out.grad = ones(2,3)
        Eigen::MatrixXd expected_grad(2,3);
        expected_grad << 0.0, 1.0, 0.0,
                         1.0, 0.0, 1.0;
        REQUIRE(a->grad.isApprox(expected_grad));
    }

    SECTION("numeric gradient check away from the kink") {
        out->backward();
        const double eps = 1e-6;
        auto loss = [](const Eigen::MatrixXd& X){
            return X.array().max(0.0).sum();   // L = sum(relu(X))
        };
        double max_err = 0.0;
        for (int i = 0; i < A.rows(); ++i)
            for (int j = 0; j < A.cols(); ++j) {
                if (A(i,j) == 0.0) continue;   // skip the kink, derivative undefined at 0
                Eigen::MatrixXd Ap = A, Am = A; Ap(i,j) += eps; Am(i,j) -= eps;
                double num = (loss(Ap) - loss(Am)) / (2*eps);
                max_err = std::max(max_err, std::abs(num - a->grad(i,j)));
            }
        REQUIRE(max_err < 1e-6);
    }
}