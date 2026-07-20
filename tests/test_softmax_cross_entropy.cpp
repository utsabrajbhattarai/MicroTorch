#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "microtorch/Tensor.hpp"
#include "microtorch/gradient_check.hpp"
#include "microtorch/ops/softmax_cross_entropy.hpp"
#include <cmath>

using namespace microtorch;

TEST_CASE("softmax_cross_entropy (unmasked overload) still behaves", "[softmax_cross_entropy]") {
    Eigen::MatrixXd Z(2,3);
    Z << 1.0, 2.0, 0.5,
         0.1, 0.2, 0.7;
    Eigen::MatrixXd Y(2,3);
    Y << 0, 1, 0,
         0, 0, 1;
    auto logits = make_tensor(Z);
    auto out    = softmax_cross_entropy(logits, Y);

    SECTION("forward is the mean cross-entropy") {
        auto rowloss = [](const Eigen::VectorXd& z, int correct){
            Eigen::VectorXd e = (z.array() - z.maxCoeff()).exp();
            Eigen::VectorXd p = e / e.sum();
            return -std::log(p(correct));
        };
        double expected = (rowloss(Z.row(0),1) + rowloss(Z.row(1),2)) / 2.0;
        REQUIRE(out->data(0,0) == Catch::Approx(expected));
    }

    SECTION("backward equals (P - Y)/N") {
        out->backward();
        Eigen::MatrixXd sh = Z.array().colwise() - Z.rowwise().maxCoeff().array();
        Eigen::MatrixXd ex = sh.array().exp();
        Eigen::MatrixXd P  = ex.array().colwise() / ex.rowwise().sum().array();
        REQUIRE(logits->grad.isApprox((P - Y) / Z.rows()));
    }
}

TEST_CASE("softmax_cross_entropy (masked overload) ignores unlabeled rows", "[softmax_cross_entropy][masked]") {
    const int N = 6, C = 2;
    Eigen::MatrixXd Z = Eigen::MatrixXd::Random(N, C);
    Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(N, C);
    Eigen::MatrixXd mask = Eigen::MatrixXd::Zero(N, 1);
    for (int i = 0; i < N; i += 2) { Y(i, i % C) = 1.0; mask(i, 0) = 1.0; }
    const double num_labeled = 3.0;

    SECTION("unlabeled rows receive exactly zero gradient") {
        auto logits = make_tensor(Z);
        auto out = softmax_cross_entropy(logits, Y, mask, num_labeled);
        out->backward();
        for (int i = 1; i < N; i += 2)
            for (int j = 0; j < C; ++j)
                REQUIRE(logits->grad(i, j) == 0.0);
    }

    SECTION("labeled rows are gradient-correct") {
        auto logits = make_tensor(Z);
        double err = gradient_check(
            [&](TensorPtr t){ return softmax_cross_entropy(t, Y, mask, num_labeled); },
            logits);
        REQUIRE(err < 1e-6);
    }

    SECTION("an all-ones mask reproduces the unmasked overload") {
        Eigen::MatrixXd Yfull = Eigen::MatrixXd::Zero(N, C);
        for (int i = 0; i < N; ++i) Yfull(i, i % C) = 1.0;
        Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(N, 1);

        auto la = make_tensor(Z);
        auto lb = make_tensor(Z);
        auto oa = softmax_cross_entropy(la, Yfull);
        auto ob = softmax_cross_entropy(lb, Yfull, ones, (double)N);
        REQUIRE(oa->data(0,0) == Catch::Approx(ob->data(0,0)));
        oa->backward(); ob->backward();
        REQUIRE(la->grad.isApprox(lb->grad));
    }
}