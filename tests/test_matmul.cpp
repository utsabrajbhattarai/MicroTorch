#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"
#include "microtorch/ops/matmul.hpp"

using namespace microtorch;

TEST_CASE("matmul: forward product and gradients", "[matmul]") {
    //A's shape is (2,3), B's is (3,2)  ->  out's is (2,2)
    Eigen::MatrixXd A(2,3);
    A << 1, 2, 3,
         4, 5, 6;
    Eigen::MatrixXd B(3,2);
    B << 7,  8,
         9, 10,
        11, 12;


    //running the operations
    auto a   = make_tensor(A);
    auto b   = make_tensor(B);
    auto out = matmul(a, b);

    SECTION("forward computes the matrix product") {
        //matmul proceeds as follows checking with hardcoded vlaue
        //row0 = [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        //row1 = [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        Eigen::MatrixXd expected(2,2);
        expected << 58,  64,
                    139, 154;
        REQUIRE(out->data.isApprox(expected));
    }

    SECTION("backward matches analytic gradients for L = sum(out)") {
        out->backward();   
        //seeds out.grad = ones, so dL/dout = ones(2,2) easy to compute and check but may need rechecking or extra carefullness because may miss some special uneven fails

        //dL/dA = ones(2,2) @ B^T
        Eigen::MatrixXd expected_a = Eigen::MatrixXd::Ones(2,2) * B.transpose();
        //dL/dB = A^T @ ones(2,2)
        Eigen::MatrixXd expected_b = A.transpose() * Eigen::MatrixXd::Ones(2,2);

        REQUIRE(a->grad.isApprox(expected_a));
        REQUIRE(b->grad.isApprox(expected_b));
    }

    //just checking the error is not much high
    SECTION("numeric gradient check (finite differences)") {
        out->backward();
        const double eps = 1e-6;
        auto loss = [](const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y){
            return (X * Y).sum();
        };
        double max_err = 0.0;
        for (int i = 0; i < A.rows(); ++i)
            for (int j = 0; j < A.cols(); ++j) {
                Eigen::MatrixXd Ap = A, Am = A; Ap(i,j) += eps; Am(i,j) -= eps;
                double num = (loss(Ap,B) - loss(Am,B)) / (2*eps);
                max_err = std::max(max_err, std::abs(num - a->grad(i,j)));
            }
        for (int i = 0; i < B.rows(); ++i)
            for (int j = 0; j < B.cols(); ++j) {
                Eigen::MatrixXd Bp = B, Bm = B; Bp(i,j) += eps; Bm(i,j) -= eps;
                double num = (loss(A,Bp) - loss(A,Bm)) / (2*eps);
                max_err = std::max(max_err, std::abs(num - b->grad(i,j)));
            }
        REQUIRE(max_err < 1e-6);
    }
}