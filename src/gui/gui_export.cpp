#include "microtorch/gui/gui_export.hpp"
#include "microtorch/eval/eval_metrics.hpp"
#include <fstream>
#include <stdexcept>
#include <filesystem>

namespace microtorch
{

// all of these are just writing to csv files, nothing else
void write_edges_csv(const std::string& path,
                      const std::vector<std::pair<int,int>>& edges) {
    std::ofstream out(path);
    if (!out.is_open()) throw std::runtime_error("could not open " + path + " for writing");

    out << "src_node_id,dst_node_id\n";
    for (const auto& edge : edges) {
        // keep the real txId1->txId2 order so the dashboard can tell sender from receiver.
        // the old src<dst filter threw away direction AND ~half the edges, which broke
        // get_neighbors()'s incoming/outgoing arrows in gnn_dashboard.
        out << edge.first << "," << edge.second << "\n";
    }
}

void write_metrics_csv(const std::string& path, const std::vector<int>& predicted,
                        const std::vector<int>& actual, const std::vector<double>& scores) {
    auto c = confusion_counts(predicted, actual);
    double p   = precision(predicted, actual);
    double r   = recall(predicted, actual);
    double f1  = f1_score(predicted, actual);
    double acc = accuracy(predicted, actual);
    double auc = auroc(scores, actual);

    std::ofstream out(path);
    if (!out.is_open()) throw std::runtime_error("could not open " + path + " for writing");

    out << "precision,recall,f1,accuracy,auroc,TP,FP,TN,FN\n";
    out << p << "," << r << "," << f1 << "," << acc << "," << auc << ","
        << c.tp << "," << c.fp << "," << c.tn << "," << c.fn << "\n";
}

void write_nodes_csv(const std::string& path, const std::vector<NodeVisData>& nodes) {
    std::ofstream out(path);
    if (!out.is_open()) throw std::runtime_error("could not open " + path + " for writing");

    out << "node_id,account_id,pred_prob,pred_label,true_label,is_test\n";
    for (const auto& node : nodes) {
        out << node.node_id << ","
            << node.account_id << ","
            << node.pred_prob << ","
            << node.pred_label << ","
            << node.true_label << ","
            << node.is_test << "\n";
    }
}

void write_accounts_csv(const std::string& path, const std::vector<AccountRecord>& accounts) {
    std::ofstream out(path);
    if (!out.is_open()) throw std::runtime_error("could not open " + path + " for writing");

    out << "rank,account_id,risk_score,num_nodes,num_illicit_pred\n";
    for (const auto& acc : accounts) {
        out << acc.rank << ","
            << acc.account_id << ","
            << acc.risk_score << ","
            << acc.num_nodes << ","
            << acc.num_illicit_pred << "\n";
    }
}

void export_all_gui_artifacts(
    const std::string& output_dir,
    const std::vector<int>& predicted,
    const std::vector<int>& actual,
    const std::vector<double>& scores,
    const std::vector<std::pair<int,int>>& edges,
    const std::vector<NodeVisData>& nodes,
    const std::vector<AccountRecord>& accounts) 
{
    // Creates gui_artifacts directory if it doesn't exist yet
    std::filesystem::create_directories(output_dir);

    write_metrics_csv(output_dir + "/metrics.csv", predicted, actual, scores);
    write_edges_csv(output_dir + "/edges.csv", edges);
    write_nodes_csv(output_dir + "/nodes.csv", nodes);
    write_accounts_csv(output_dir + "/accounts.csv", accounts);
}

} //namespace microtorch