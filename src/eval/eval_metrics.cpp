// src/eval/eval_metrics.cpp
#include "microtorch/eval/eval_metrics.hpp"
#include <algorithm>   // for std::sort
#include <stdexcept>  // for std::invalid_argument, std::runtime_error etc
#include <fstream>

ConfusionCounts confusion_counts(const std::vector<int>& predicted, const std::vector<int>& actual) {
    ConfusionCounts c{0, 0, 0, 0};
    for (size_t i = 0; i < predicted.size(); ++i) {
        if (predicted[i] == 1 && actual[i] == 1) c.tp++; // true positive(fraud correctly predicted)
        else if (predicted[i] == 1 && actual[i] == 0) c.fp++; // false positive(clean incorrectly predicted as fraud)
        else if (predicted[i] == 0 && actual[i] == 0) c.tn++; // true negative(clean correctly predicted)
        else if (predicted[i] == 0 && actual[i] == 1) c.fn++; // false negative(fraud incorrectly predicted as clean)
    }
    return c;
}

// positive : fraud, negative : clean

// precision : out of all predicted positives, how many were actually positive

double precision(const std::vector<int>& predicted, const std::vector<int>& actual) {
    auto c = confusion_counts(predicted, actual);
    if (c.tp + c.fp == 0) return 0.0;
    return static_cast<double>(c.tp) / (c.tp + c.fp);
}

// recall : out of all actual positives, how many were predicted positive

double recall(const std::vector<int>& predicted, const std::vector<int>& actual) {
    auto c = confusion_counts(predicted, actual);
    if (c.tp + c.fn == 0) return 0.0;
    return static_cast<double>(c.tp) / (c.tp + c.fn);
}


// f1_score : harmonic mean of precision and recall
double f1_score(const std::vector<int>& predicted, const std::vector<int>& actual) {
    double p = precision(predicted, actual);
    double r = recall(predicted, actual);
    if (p + r == 0) return 0.0;
    return 2 * p * r / (p + r);
}

double accuracy(const std::vector<int>& predicted, const std::vector<int>& actual) {
    auto c = confusion_counts(predicted, actual);
    int total = c.tp + c.fp + c.tn + c.fn;
    if (total == 0) return 0.0;
    return static_cast<double>(c.tp + c.tn) / total;
}

double auroc(const std::vector<double>& scores, const std::vector<int>& actual) {
    size_t n = scores.size();
    if (n != actual.size()) throw std::invalid_argument("size mismatch");

    // pair each score with its label, so sorting keeps the label attached
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
        double avg_rank = (i + 1 + j) / 2.0; // average of ranks for equal scores
        for (size_t k = i; k < j; ++k) ranks[k] = avg_rank;
        i = j;
    }

    double rank_sum_pos = 0.0;
    int n_pos = 0, n_neg = 0;
    for (size_t k = 0; k < n; ++k) {
        if (paired[k].second == 1) { rank_sum_pos += ranks[k]; n_pos++; }
        else n_neg++;
    }

    if (n_pos == 0 || n_neg == 0)
        throw std::invalid_argument("AUROC needs both classes present");

    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg);
}

