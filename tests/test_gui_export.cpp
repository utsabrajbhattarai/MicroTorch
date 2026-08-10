#include <catch2/catch_test_macros.hpp>
#include "microtorch/gui/gui_export.hpp"
#include <fstream>
#include <filesystem>

TEST_CASE("write_metrics_csv produces valid CSV format", "[gui_export]") {
    std::string path = "test_metrics.csv";
    std::vector<int> predicted = {1, 1, 0, 0};
    std::vector<int> actual    = {1, 0, 1, 0};
    std::vector<double> scores = {0.9, 0.8, 0.3, 0.1};

    write_metrics_csv(path, predicted, actual, scores);

    std::ifstream file(path);
    REQUIRE(file.is_open());

    std::string header, data_row;
    std::getline(file, header);
    std::getline(file, data_row);

    REQUIRE(header == "precision,recall,f1,accuracy,auroc,TP,FP,TN,FN");
    REQUIRE_FALSE(data_row.empty());

    file.close();
    std::filesystem::remove(path);
}

TEST_CASE("write_edges_csv filters self-loops and backward edges", "[gui_export]") {
    std::string path = "test_edges.csv";
    std::vector<std::pair<int,int>> edges = {
        {0, 1},
        {1, 0},
        {2, 3},
        {2, 2}
    };

    write_edges_csv(path, edges);

    std::ifstream file(path);
    REQUIRE(file.is_open());

    std::string header;
    std::getline(file, header);
    REQUIRE(header == "src_node_id,dst_node_id");

    std::vector<std::string> rows;
    std::string line;
    while (std::getline(file, line)) {
        rows.push_back(line);
    }

    REQUIRE(rows.size() == 2);
    REQUIRE(rows[0] == "0,1");
    REQUIRE(rows[1] == "2,3");

    file.close();
    std::filesystem::remove(path);
}

TEST_CASE("write_nodes_csv matches gui contract", "[gui_export]") {
    std::string path = "test_nodes.csv";
    std::vector<NodeVisData> nodes = {
        {0, "27238738", 0.91, 1, 1, 1},
        {1, "88123401", 0.04, 0, 0, 1}
    };

    write_nodes_csv(path, nodes);

    std::ifstream file(path);
    REQUIRE(file.is_open());

    std::string header, line1, line2;
    std::getline(file, header);
    std::getline(file, line1);
    std::getline(file, line2);

    REQUIRE(header == "node_id,account_id,pred_prob,pred_label,true_label,is_test");
    REQUIRE(line1 == "0,27238738,0.91,1,1,1");
    REQUIRE(line2 == "1,88123401,0.04,0,0,1");

    file.close();
    std::filesystem::remove(path);
}

TEST_CASE("write_accounts_csv matches gui contract", "[gui_export]") {
    std::string path = "test_accounts.csv";
    std::vector<AccountRecord> accounts = {
        {1, "27238738", 0.94, 12, 7},
        {2, "88123401", 0.88, 5, 3}
    };

    write_accounts_csv(path, accounts);

    std::ifstream file(path);
    REQUIRE(file.is_open());

    std::string header, line1;
    std::getline(file, header);
    std::getline(file, line1);

    REQUIRE(header == "rank,account_id,risk_score,num_nodes,num_illicit_pred");
    REQUIRE(line1 == "1,27238738,0.94,12,7");

    file.close();
    std::filesystem::remove(path);
}