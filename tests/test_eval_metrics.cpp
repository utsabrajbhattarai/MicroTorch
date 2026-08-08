#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "microtorch/eval/eval_metrics.hpp"


TEST_CASE("confusion_counts on hand-checked example") {
    std::vector<int> predicted = {1, 1, 0, 0};
    std::vector<int> actual    = {1, 0, 1, 0};

    ConfusionCounts c = confusion_counts(predicted, actual);

    REQUIRE(c.tp == 1);  // index 0: predicted 1, actual 1
    REQUIRE(c.fp == 1);  // index 1: predicted 1, actual 0
    REQUIRE(c.fn == 1);  // index 2: predicted 0, actual 1
    REQUIRE(c.tn == 1);  // index 3: predicted 0, actual 0
}

TEST_CASE("f1_score on hand-checked example") {
    std::vector<int> predicted = {1, 1, 0, 0};
    std::vector<int> actual    = {1, 0, 1, 0};

    // precision = 0.5, recall = 0.5
    // f1 = 2*0.5*0.5 / (0.5+0.5) = 0.5 / 1.0 = 0.5
    REQUIRE(f1_score(predicted, actual) == Catch::Approx(0.5));
}

TEST_CASE("f1_score edge case: precision and recall both zero") {
    std::vector<int> predicted = {0, 0, 0};
    std::vector<int> actual    = {1, 1, 1}; // model catches nothing

    REQUIRE(f1_score(predicted, actual) == Catch::Approx(0.0));
}

TEST_CASE("f1_score perfect predictions") {
    std::vector<int> predicted = {1, 0, 1, 0};
    std::vector<int> actual    = {1, 0, 1, 0};

    REQUIRE(f1_score(predicted, actual) == Catch::Approx(1.0));
}

TEST_CASE("accuracy on hand-checked example") {
    std::vector<int> predicted = {1, 1, 0, 0};
    std::vector<int> actual    = {1, 0, 1, 0};

    // tp=1, tn=1, total=4 -> (1+1)/4 = 0.5
    REQUIRE(accuracy(predicted, actual) == Catch::Approx(0.5));
}

TEST_CASE("accuracy perfect and worst cases") {
    std::vector<int> predicted_perfect = {1, 0, 1, 0};
    std::vector<int> actual_perfect    = {1, 0, 1, 0};
    REQUIRE(accuracy(predicted_perfect, actual_perfect) == Catch::Approx(1.0));

    std::vector<int> predicted_worst = {1, 0, 1, 0};
    std::vector<int> actual_worst    = {0, 1, 0, 1}; // exactly backwards
    REQUIRE(accuracy(predicted_worst, actual_worst) == Catch::Approx(0.0));
}

