#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"
#include "microtorch/gradient_check.hpp"
#include "microtorch/gnn/gnn_model.hpp"
#include "microtorch/ops/softmax_cross_entropy.hpp"
#include <vector>

using namespace microtorch;

TEST_CASE("GNNModel: full forward/backward is gradient-correct", "[gnn]") {
    const int N = 6, F = 5, C = 2;

    // small hand-built graph
    Eigen::SparseMatrix<double> A(N, N);
    std::vector<Eigen::Triplet<double>> t = {
        {0,0,1},{0,1,1},{1,1,1},{1,2,1},{2,2,1},{2,3,1},
        {3,3,1},{3,4,1},{4,4,1},{4,5,1},{5,5,1},{5,0,1}
    };
    A.setFromTriplets(t.begin(), t.end());

    Eigen::MatrixXd Xd = Eigen::MatrixXd::Random(N, F);
    Eigen::MatrixXd Y  = Eigen::MatrixXd::Zero(N, C);
    for (int i = 0; i < N; ++i) Y(i, i % C) = 1.0;   //one-hot labels

    GNNModel model(F, 8, C);
    auto x = make_tensor(Xd);

    //wrap the whole network + loss as one (tensor)->scalar function
    double err = gradient_check(
        [&](TensorPtr in){ return softmax_cross_entropy(model.forward(A, in), Y); },
        x);

    REQUIRE(err < 1e-6);
}