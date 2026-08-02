
#pragma once
#include <vector>

double precision(const std::vector<int>& predicted, const std::vector<int>& actual); 
double recall(const std::vector<int>& predicted, const std::vector<int>& actual); 
double auroc(const std::vector<double>& scores, const std::vector<int>& actual); 