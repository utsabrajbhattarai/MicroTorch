#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"
#include "microtorch/ops/neighbor_aggregation.hpp"
#include <vector>

using namespace microtorch;

TEST_CASE("neighbor_aggregation: forward is A*X, backward is A^T * grad", "[neighbor_aggregation]") {
    const int N = 3, F = 2;

    // Build a small 3-node sparse adjacency by hand.
    // Triplet is (row, col, value); setFromTriplets assembles them into the sparse matrix.
    Eigen::SparseMatrix<double> A(N, N);
    std::vector<Eigen::Triplet<double>> trips = {
        {0,0,1.0}, {0,1,1.0},   // node 0 <- itself and node 1
        {1,0,1.0}, {1,2,1.0},   // node 1 <- node 0 and node 2
        {2,1,1.0}                // node 2 <- node 1
    };
    A.setFromTriplets(trips.begin(), trips.end());

    Eigen::MatrixXd X(N, F);
    X << 1, 2,
         3, 4,
         5, 6;

    auto x   = make_tensor(X);
    auto out = neighbor_aggregation(A, x);

    SECTION("forward sums the right neighbor rows") {
        // node 0 = row0 + row1 = [4,6]; node 1 = row0 + row2 = [6,8]; node 2 = row1 = [3,4]
        Eigen::MatrixXd expected(N, F);
        expected << 4, 6,
                    6, 8,
                    3, 4;
        REQUIRE(out->data.isApprox(expected));
    }

    SECTION("backward matches A^T * upstream and numeric check") {
        out->backward();

        Eigen::MatrixXd Adense = Eigen::MatrixXd(A);
        Eigen::MatrixXd expected_grad = Adense.transpose() * Eigen::MatrixXd::Ones(N, F);
        REQUIRE(x->grad.isApprox(expected_grad));

        const double eps = 1e-6; double max_err = 0.0;
        auto loss = [&](const Eigen::MatrixXd& M){ return (Adense * M).sum(); };
        for (int i=0;i<N;++i) for (int j=0;j<F;++j){
            Eigen::MatrixXd Xp=X,Xm=X; Xp(i,j)+=eps; Xm(i,j)-=eps;
            double num = (loss(Xp)-loss(Xm))/(2*eps);
            max_err = std::max(max_err, std::abs(num - x->grad(i,j)));
        }
        REQUIRE(max_err < 1e-6);
    }
}