// src/eval/eval_metrics.cpp
#include "microtorch/eval/eval_metrics.hpp"
#include <algorithm>
#include <stdexcept>


// precision : of all predicted positives(fraud ones), how many are actually positive(fraud)
double precision(const std::vector<int>& predicted, const std::vector<int>& actual) {
    int tp = 0, fp = 0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        if (predicted[i] == 1 && actual[i] == 1) tp++;  // 1 is index for fraud. so true positive is when predicted is 1 and actual is also 1
        else if (predicted[i] == 1 && actual[i] == 0) fp++;  //false positive is when predicted is 1 but actual is 0
    }
    if (tp + fp == 0) return 0.0; // no predicted positives at all
    return static_cast<double>(tp) / (tp + fp);  // tp and fp are ints so casting to double for fraction value
}


// recall : of all actual positives, how many are predicted positives
double recall(const std::vector<int>& predicted, const std::vector<int>& actual) {
    int tp = 0, fn = 0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        if (predicted[i] == 1 && actual[i] == 1) tp++;
        else if (predicted[i] == 0 && actual[i] == 1) fn++;
    }
    if (tp + fn == 0) return 0.0; // no actual positives at all
    return static_cast<double>(tp) / (tp + fn);
}

double auroc(const std::vector<double>& scores, const std::vector<int>& actual) {
    size_t n = scores.size();
    if (n != actual.size()) throw std::invalid_argument("size mismatch");

    // pairing each score with its label, so sorting keeps the label attached
    std::vector<std::pair<double,int>> paired(n);
    for (size_t i = 0; i < n; ++i) paired[i] = {scores[i], actual[i]};

    // sorting in ascending order of scores, so that we can assign ranks
    std::sort(paired.begin(), paired.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // assign ranks (1-indexed), averaging ranks within tie blocks
    std::vector<double> ranks(n);
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        while (j < n && paired[j].first == paired[i].first) j++; // find tie block [i, j)
        double avg_rank = (i + 1 + j) / 2.0; // average of ranks i+1 .. j
        for (size_t k = i; k < j; ++k) ranks[k] = avg_rank;
        i = j;
    }

    double rank_sum_pos = 0.0;  // sum of ranks for positive instances(fraud cases)
    int n_pos = 0, n_neg = 0;  // n_pos = no of positive cases, n_neg = no of negative cases
    for (size_t k = 0; k < n; ++k) {
        if (paired[k].second == 1) { rank_sum_pos += ranks[k]; n_pos++; }
        else n_neg++;
    }

    if (n_pos == 0 || n_neg == 0)
        throw std::invalid_argument("AUROC needs both classes present");

    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg);
}