#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "microtorch/diffusion/diffusion_model.hpp"
#include "microtorch/gradient_check.hpp"

using namespace microtorch;

TEST_CASE("forward_noise: t=0 preserves the point, large t destroys it", "[diffusion][forward_noise]") {
    NoiseSchedule ns = make_noise_schedule(200);
    std::mt19937 rng(1);

    Eigen::MatrixXd x0(3, 2);
    x0 << 1.0, 1.0,  -2.0, 0.5,  0.0, -1.5;
    Eigen::MatrixXd noise;

    SECTION("output and noise share x0's shape") {
        Eigen::MatrixXd xt = forward_noise(x0, 50, ns, noise, rng);
        REQUIRE(xt.rows()    == x0.rows());
        REQUIRE(xt.cols()    == x0.cols());
        REQUIRE(noise.rows() == x0.rows());
        REQUIRE(noise.cols() == x0.cols());
    }

    SECTION("at t=0 the noised point is almost identical to x0") {
        Eigen::MatrixXd xt = forward_noise(x0, 0, ns, noise, rng);
        REQUIRE((xt - x0).cwiseAbs().maxCoeff() < 0.1);   // barely moved
    }

    SECTION("at large t the point is dominated by noise, not x0") {
        Eigen::MatrixXd xt = forward_noise(x0, 199, ns, noise, rng);
        // signal coefficient is small, so x_t should be far from x0 on average
        REQUIRE((xt - x0).cwiseAbs().mean() > 0.3);
    }

    SECTION("the blend obeys x_t = sqrt(abar)*x0 + sqrt(1-abar)*noise") {
        int t = 75;
        Eigen::MatrixXd xt = forward_noise(x0, t, ns, noise, rng);
        double c1 = std::sqrt(ns.alpha_bar[t]);
        double c2 = std::sqrt(1.0 - ns.alpha_bar[t]);
        REQUIRE(xt.isApprox(c1 * x0 + c2 * noise));   // reconstruct from the returned noise
    }
}


TEST_CASE("DiffusionModel: full denoiser forward/backward is gradient-correct", "[diffusion][model]") {
    DiffusionModel model(3, 8, 2);                       // in=3 (x,y,t), hidden=8, out=2 (noise)
    Eigen::MatrixXd inp    = Eigen::MatrixXd::Random(5, 3);
    Eigen::MatrixXd target = Eigen::MatrixXd::Random(5, 2);   // stand-in "true noise"
    auto x = make_tensor(inp);

    double err = gradient_check(
        [&](TensorPtr t){ return mse_loss(model.forward(t), target); },
        x);
    REQUIRE(err < 1e-6);
}