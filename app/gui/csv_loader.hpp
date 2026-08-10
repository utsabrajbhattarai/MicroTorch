#pragma once
#include <string>
#include <vector>

//All structs represent the data in the CSV files. The field names match the column names in the CSV files.

struct Node {
    int node_id;
    long long account_id;
    double pred_prob;
    int pred_label;
    int true_label;
    int is_test;
};

struct Edge {
    int src_node_id;
    int dst_node_id;
};

struct Account {
    int rank;
    long long account_id;
    double risk_score;
    int num_nodes;
    int num_illicit_pred;
};

struct Metrics {
    double precision;
    double recall;
    double f1;
    double accuracy;
    double auroc;

    /*Area Under the Receiver Operating Characteristic curve
    AUROC is a performance measurement for classification problems at various threshold settings.
    It tells how much the model is capable of distinguishing between classes.
    The higher the AUROC, the better the model is at predicting 0s as 0s and 1s as 1s.
    The AUROC is calculated by plotting the true positive rate (TPR) against the false positive rate (FPR) 
    at various threshold settings.*/

    int TP, FP, TN, FN; //true positives, false positives, true negatives, false negatives
};

//Function declarations for loading CSV files. The definitions of these functions are in csv_loader.cpp
//                                          |
//                                          v
std::vector<Node> load_nodes_csv(const std::string& filepath); 
std::vector<Edge> load_edges_csv(const std::string& filepath);
std::vector<Account> load_accounts_csv(const std::string& filepath);
Metrics load_metrics_csv(const std::string& filepath);