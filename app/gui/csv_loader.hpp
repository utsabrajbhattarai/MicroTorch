#pragma once
#include <string>
#include <vector>

struct Node { //loads node data from csv file
    int node_id;
    long long account_id;
    double pred_prob;
    int pred_label;
    int true_label;
    int is_test;
};

std::vector<Node> load_nodes_csv(const std::string& filepath); //vector of nodes loaded from csv file