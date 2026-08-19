#pragma once
#include <string>
#include <vector>
#include <utility>

namespace microtorch{
    
// for displaying nodes in gui, exporting the following things
struct NodeVisData {
    int node_id;
    std::string account_id;
    double pred_prob;
    int pred_label;
    int true_label;
    int is_test;
};

// accounts.csv necessary terms
struct AccountRecord {
    int rank;
    std::string account_id;
    double risk_score;
    int num_nodes;
    int num_illicit_pred;
};


void write_metrics_csv(const std::string& path, const std::vector<int>& predicted, 
                       const std::vector<int>& actual, const std::vector<double>& scores);

void write_edges_csv(const std::string& path, const std::vector<std::pair<int,int>>& edges);

void write_nodes_csv(const std::string& path, const std::vector<NodeVisData>& nodes);

void write_accounts_csv(const std::string& path, const std::vector<AccountRecord>& accounts);

//master function to export all gui artifacts
void export_all_gui_artifacts(
    const std::string& output_dir,
    const std::vector<int>& predicted,
    const std::vector<int>& actual,
    const std::vector<double>& scores,
    const std::vector<std::pair<int,int>>& edges,
    const std::vector<NodeVisData>& nodes,
    const std::vector<AccountRecord>& accounts
);

} //namespace microtorch