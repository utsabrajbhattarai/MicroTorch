// tests/test_eval_metrics.cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "microtorch/eval/eval_metrics.hpp"

TEST_CASE("precision and recall on hand-checked example") {
    std::vector<int> predicted = {1, 1, 0, 0};
    std::vector<int> actual    = {1, 0, 1, 0};

    REQUIRE(precision(predicted, actual) == Catch::Approx(0.5));
    REQUIRE(recall(predicted, actual) == Catch::Approx(0.5));
}

TEST_CASE("precision and recall edge cases") {
    // no positive predictions at all -> precision defined as 0.0
    std::vector<int> predicted_none = {0, 0, 0};
    std::vector<int> actual_none    = {1, 0, 1};
    REQUIRE(precision(predicted_none, actual_none) == Catch::Approx(0.0));

    // no actual positives at all -> recall defined as 0.0
    std::vector<int> predicted_2 = {1, 0, 0};
    std::vector<int> actual_2    = {0, 0, 0};
    REQUIRE(recall(predicted_2, actual_2) == Catch::Approx(0.0));

    // perfect predictions -> both should be 1.0
    std::vector<int> predicted_perfect = {1, 0, 1, 0};
    std::vector<int> actual_perfect    = {1, 0, 1, 0};
    REQUIRE(precision(predicted_perfect, actual_perfect) == Catch::Approx(1.0));
    REQUIRE(recall(predicted_perfect, actual_perfect) == Catch::Approx(1.0));
}

TEST_CASE("auroc on hand-checked example") {
    std::vector<double> scores = {0.9, 0.8, 0.3, 0.1};
    std::vector<int> actual    = {1,   0,   1,   0};

    REQUIRE(auroc(scores, actual) == Catch::Approx(0.75));
}

TEST_CASE("auroc perfect and random cases") {
    // perfect separation: all positives score higher than all negatives
    std::vector<double> scores_perfect = {0.9, 0.8, 0.2, 0.1};
    std::vector<int> actual_perfect    = {1,   1,   0,   0};
    REQUIRE(auroc(scores_perfect, actual_perfect) == Catch::Approx(1.0));

    // worst case: model has it exactly backwards
    std::vector<double> scores_backwards = {0.1, 0.2, 0.8, 0.9};
    std::vector<int> actual_backwards    = {1,   1,   0,   0};
    REQUIRE(auroc(scores_backwards, actual_backwards) == Catch::Approx(0.0));
}

TEST_CASE("auroc with tied scores") {
    // two samples tied at 0.5: one fraud, one clean
    // sorted: 0.1(clean,rank1), 0.5(clean,rank2.5), 0.5(fraud,rank2.5), 0.9(fraud,rank4)
    std::vector<double> scores = {0.9, 0.1, 0.5, 0.5};
    std::vector<int> actual    = {1,   0,   0,   1};

    // rank_sum_pos = 4 (for 0.9) + 2.5 (for the tied 0.5) = 6.5
    // n_pos = 2, n_neg = 2
    // AUROC = (6.5 - 2*3/2) / (2*2) = (6.5 - 3) / 4 = 3.5 / 4 = 0.875
    REQUIRE(auroc(scores, actual) == Catch::Approx(0.875));
}

TEST_CASE("auroc throws on mismatched sizes") {
    std::vector<double> scores = {0.9, 0.1};
    std::vector<int> actual    = {1, 0, 1};

    REQUIRE_THROWS_AS(auroc(scores, actual), std::invalid_argument);
}

TEST_CASE("auroc throws when only one class present") {
    std::vector<double> scores = {0.9, 0.5, 0.1};
    std::vector<int> actual    = {1, 1, 1}; // no negatives at all

    REQUIRE_THROWS_AS(auroc(scores, actual), std::invalid_argument);
}