#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"
#include "microtorch/ops/broadcast_add.hpp"

using namespace microtorch;

TEST_CASE("broadcast_add: bias add and its gradients", "[broadcast_add]") {
    Eigen::MatrixXd A(3,4);
    A << 1, 2, 3, 4,
         5, 6, 7, 8,
         9,10,11,12;
    Eigen::MatrixXd Bv(1,4);
    Bv << 100, 200, 300, 400;
    //checking if the bradcast_add correctly does 3X4 + 1x4 dummy testing

    //just making them tensors using eigen values
    auto a   = make_tensor(A);
    auto b   = make_tensor(Bv);
    auto out = broadcast_add(a, b);

    //hardcoding the expected output
    SECTION("forward adds the bias row to every row") {
        Eigen::MatrixXd expected(3,4);
        expected << 101, 202, 303, 404,
                    105, 206, 307, 408,
                    109, 210, 311, 412;
        REQUIRE(out->data.isApprox(expected));  //is generated output approximately close to the actual 
    }


    //checking for the backward's working
    SECTION("backward produces the correct analytic gradients") {
        out->backward();
        REQUIRE(a->grad.isApprox(Eigen::MatrixXd::Ones(3,4)));
        Eigen::MatrixXd expected_b(1,4);
        expected_b.setConstant(3.0);
        REQUIRE(b->grad.isApprox(expected_b));
    }
}