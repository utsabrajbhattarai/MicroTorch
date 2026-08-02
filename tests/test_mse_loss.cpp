#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "microtorch/Tensor.hpp"
#include "microtorch/ops/mse_loss.hpp"

using namespace microtorch;

TEST_CASE("mse_loss: forward error and gradient", "[mse_loss]") {
    Eigen::MatrixXd P(2,2);
    P << 1, 2,
         3, 4;
    Eigen::MatrixXd T(2,2);
    T << 0, 2,
         3, 0;
    // diff = [[1,0],[0,4]], squared sum = 17, N = 4, MSE = 4.25

    auto pred = make_tensor(P);
    auto out  = mse_loss(pred, T);

    SECTION("forward is mean of squared differences") {
        REQUIRE(out->data(0,0) == Catch::Approx(4.25));
    }

    SECTION("backward is (2/N)*(pred - target)") {
        out->backward();
        Eigen::MatrixXd expected(2,2);
        expected << 0.5, 0.0,
                    0.0, 2.0;
        REQUIRE(pred->grad.isApprox(expected));
    }

    SECTION("numeric gradient check") {
        out->backward();
        const double eps = 1e-6; double max_err = 0.0;
        auto loss = [&](const Eigen::MatrixXd& X){ return (X - T).squaredNorm() / X.size(); };
        for (int i=0;i<P.rows();++i) for (int j=0;j<P.cols();++j){
            Eigen::MatrixXd Pp=P,Pm=P; Pp(i,j)+=eps; Pm(i,j)-=eps;
            double num = (loss(Pp) - loss(Pm)) / (2*eps);
            max_err = std::max(max_err, std::abs(num - pred->grad(i,j)));
        }
        REQUIRE(max_err < 1e-6);
    }
}