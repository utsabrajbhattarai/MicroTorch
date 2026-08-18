#pragma once  // if this header file is included multiple times, it will only be included once, avoiding redefinition errors
#include <Eigen/Dense>  //for Eigen::MatrixXd
#include <string>     // for std::string
#include <vector>  // for vector
#include <unordered_map> // for unordered_map
#include <utility> // for std::pair
#include <Eigen/Sparse>  // for sparse matrix- matrix having tons of 0s (no connections for our edgelist file)

// created microtorch namespace including load_csv function 
namespace microtorch{
    Eigen::MatrixXd load_csv(const std::string& path, bool has_header = false);  // function declaration for loading CSV files into Eigen::MatrixXd

    std::vector <int> load_labels(const std::string& path, bool has_header = false); // returns a vector of labels corresponding to the transactions in csv file, 1 for fraud, 0 for legitimate, -1 for unknown

    std::unordered_map<long long, int> build_txid_index(const std::string& path, bool has_header = false); //maps the id of transaction i.e long long txid to its corresponding row index in the csv file, returns an unordered_map of txid to row index

    std::vector<std::pair<int,int>> load_edges(const std::string& path, const std::unordered_map<long long, int>& txid_to_row, bool has_header = false);

    struct EllipticData {
    Eigen::SparseMatrix<double> A;
    Eigen::MatrixXd X;
    Eigen::MatrixXd Y;
    std::vector<int> labeled_rows;
    std::vector<std::string> tx_ids;    // row index -> original txid, so the gui can label a node by its real account
    std::vector<std::pair<int, int>> raw_edges;  // pre-symmetrisation edges, keeps the real sender->receiver direction A throws away
    };

    EllipticData load_elliptic(const std::string& features_path, const std::string& classes_path, const std::string& edges_path);

    
}