#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "microtorch/Tensor.hpp"
#include "microtorch/ops/softmax_cross_entropy.hpp"
#include <cmath>

using namespace microtorch;

TEST_CASE("softmax_cross_entropy: forward loss and P-Y gradient", "[softmax_cross_entropy]") {
    Eigen::MatrixXd Z(2,3);
    Z << 1.0, 2.0, 0.5,
         0.1, 0.2, 0.7;
    Eigen::MatrixXd Y(2,3);
    Y << 0, 1, 0,
         0, 0, 1;

    auto logits = make_tensor(Z);
    auto out    = softmax_cross_entropy(logits, Y);

    SECTION("forward is the mean cross-entropy of the softmax") {
        auto rowloss = [](const Eigen::VectorXd& z, int correct){
            Eigen::VectorXd e = (z.array() - z.maxCoeff()).exp();
            Eigen::VectorXd p = e / e.sum();
            return -std::log(p(correct));
        };
        double expected = (rowloss(Z.row(0), 1) + rowloss(Z.row(1), 2)) / 2.0;
        REQUIRE(out->data(0,0) == Catch::Approx(expected));
    }

    SECTION("backward equals (softmax(logits) - targets)/N") {
        out->backward();
        Eigen::MatrixXd sh = Z.array().colwise() - Z.rowwise().maxCoeff().array();
        Eigen::MatrixXd ex = sh.array().exp();
        Eigen::MatrixXd P  = ex.array().colwise() / ex.rowwise().sum().array();
        Eigen::MatrixXd expected_grad = (P - Y) / Z.rows();
        REQUIRE(logits->grad.isApprox(expected_grad));
    }

    SECTION("numeric gradient check") {
        out->backward();
        auto loss_of = [](const Eigen::MatrixXd& z, const Eigen::MatrixXd& y){
            Eigen::MatrixXd sh = z.array().colwise() - z.rowwise().maxCoeff().array();
            Eigen::MatrixXd ex = sh.array().exp();
            Eigen::MatrixXd p  = ex.array().colwise() / ex.rowwise().sum().array();
            return -(y.array() * p.array().log()).sum() / p.rows();
        };
        const double eps = 1e-6; double max_err = 0.0;
        for (int i=0;i<Z.rows();++i) for (int j=0;j<Z.cols();++j){
            Eigen::MatrixXd Zp=Z,Zm=Z; Zp(i,j)+=eps; Zm(i,j)-=eps;
            double num = (loss_of(Zp,Y) - loss_of(Zm,Y)) / (2*eps);
            max_err = std::max(max_err, std::abs(num - logits->grad(i,j)));
        }
        REQUIRE(max_err < 1e-6);
    }
}