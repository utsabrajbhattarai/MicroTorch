
#pragma once
#include <vector>
#include <string>

struct ConfusionCounts
{
    int tp, fp, tn, fn;
};

ConfusionCounts confusion_counts(const std::vector<int>& predicted, const std::vector<int>& actual);

double precision(const std::vector<int>& predicted, const std::vector<int>& actual); 
double recall(const std::vector<int>& predicted, const std::vector<int>& actual); 
double f1_score(const std::vector<int>& predicted, const std::vector<int>& actual);
double accuracy(const std::vector<int>& predicted, const std::vector<int>& actual);
double auroc(const std::vector<double>& scores, const std::vector<int>& actual); 

