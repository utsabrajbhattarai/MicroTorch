#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "microtorch/Tensor.hpp"
#include "microtorch/ops/reduction.hpp"

using namespace microtorch;

TEST_CASE("sum: forward total and broadcast-back gradient", "[reduction][sum]") {
    Eigen::MatrixXd A(2,3);
    A << 1, 2, 3,
         4, 5, 6;                       // sum = 21

    auto a   = make_tensor(A);
    auto out = sum(a);

    SECTION("forward is the total") {
        REQUIRE(out->data(0,0) == Catch::Approx(21.0));
    }
    SECTION("backward spreads grad unchanged to every cell") {
        out->backward();
        REQUIRE(a->grad.isApprox(Eigen::MatrixXd::Ones(2,3)));
    }
    SECTION("numeric gradient check") {
        out->backward();
        const double eps = 1e-6; double max_err = 0.0;
        for (int i=0;i<A.rows();++i) for (int j=0;j<A.cols();++j){
            Eigen::MatrixXd Ap=A,Am=A; Ap(i,j)+=eps; Am(i,j)-=eps;
            double num = (Ap.sum() - Am.sum()) / (2*eps);
            max_err = std::max(max_err, std::abs(num - a->grad(i,j)));
        }
        REQUIRE(max_err < 1e-6);
    }
}

TEST_CASE("mean: forward average and scaled broadcast-back gradient", "[reduction][mean]") {
    Eigen::MatrixXd A(2,3);
    A << 1, 2, 3,
         4, 5, 6;                       // mean = 3.5, size = 6

    auto a   = make_tensor(A);
    auto out = mean(a);

    SECTION("forward is the average") {
        REQUIRE(out->data(0,0) == Catch::Approx(3.5));
    }
    SECTION("backward spreads 1/size to every cell") {
        out->backward();
        REQUIRE(a->grad.isApprox(Eigen::MatrixXd::Constant(2,3,1.0/6.0)));
    }
    SECTION("numeric gradient check") {
        out->backward();
        const double eps = 1e-6; double max_err = 0.0;
        for (int i=0;i<A.rows();++i) for (int j=0;j<A.cols();++j){
            Eigen::MatrixXd Ap=A,Am=A; Ap(i,j)+=eps; Am(i,j)-=eps;
            double num = (Ap.mean() - Am.mean()) / (2*eps);
            max_err = std::max(max_err, std::abs(num - a->grad(i,j)));
        }
        REQUIRE(max_err < 1e-6);
    }
}