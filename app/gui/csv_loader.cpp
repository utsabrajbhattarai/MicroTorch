#include "csv_loader.hpp"
#include <fstream>
#include <sstream>

std::vector<Node> load_nodes_csv(const std::string& filepath) { //vector of nodes loaded from csv file
    std::vector<Node> nodes; //node data loaded from csv file
    std::ifstream file(filepath); //node data loaded from csv file
    std::string line; //node data loaded from csv file

    std::getline(file, line);  //Skip the header line

    while (std::getline(file, line)) { //Read each line of the CSV file
        std::stringstream ss(line); 
        std::string field; //field to hold each value in the line
        Node n; //initialize a new Node object to hold the data for this line

        std::getline(ss, field, ','); n.node_id = std::stoi(field);
        std::getline(ss, field, ','); n.account_id = std::stoll(field);
        std::getline(ss, field, ','); n.pred_prob = std::stod(field);
        std::getline(ss, field, ','); n.pred_label = std::stoi(field);
        std::getline(ss, field, ','); n.true_label = std::stoi(field);
        std::getline(ss, field, ','); n.is_test = std::stoi(field);

        nodes.push_back(n); //Add the Node object to the vector of nodes
    }

    return nodes;
}

//Function definitions for loading edges, accounts, and metrics CSV files; from csv_loader.hpp
std::vector<Edge> load_edges_csv(const std::string& filepath) {
    std::vector<Edge> edges; //vector of edges loaded from csv file
    std::ifstream file(filepath); //input file stream to read the CSV file
    std::string line;

    std::getline(file, line);  //skip header

    while (std::getline(file, line)) { //Read each line of the CSV file
        std::stringstream ss(line); //line is split into fields using a stringstream
        std::string field; //field to hold each value in the line
        Edge e; //initialize a new Edge object to hold the data for this line

        //fields are read from the stringstream and converted to the appropriate data type before being assigned to the Edge object

        std::getline(ss, field, ','); e.src_node_id = std::stoi(field); 
        std::getline(ss, field, ','); e.dst_node_id = std::stoi(field);

        edges.push_back(e); //Add the Edge object to the vector of edges
    }

    return edges;
}

std::vector<Account> load_accounts_csv(const std::string& filepath) {
    std::vector<Account> accounts;
    std::ifstream file(filepath);
    std::string line;

    std::getline(file, line);  //skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        Account a;

        std::getline(ss, field, ','); a.rank = std::stoi(field);
        std::getline(ss, field, ','); a.account_id = std::stoll(field);
        std::getline(ss, field, ','); a.risk_score = std::stod(field);
        std::getline(ss, field, ','); a.num_nodes = std::stoi(field);
        std::getline(ss, field, ','); a.num_illicit_pred = std::stoi(field);

        accounts.push_back(a);
    }

    return accounts;
}

Metrics load_metrics_csv(const std::string& filepath) {
    Metrics m;
    std::ifstream file(filepath);
    std::string line;

    std::getline(file, line);  //skip header
    std::getline(file, line);  //the one data row

    std::stringstream ss(line);
    std::string field;

    //fields are read from the stringstream and converted to the appropriate data type before being assigned to the Metrics object
    
    //convert the string values to the appropriate data types (double or int) and assign them to the corresponding fields in the Metrics struct
    //                          |
    //                          v
    std::getline(ss, field, ','); m.precision = std::stod(field); 
    std::getline(ss, field, ','); m.recall = std::stod(field);
    std::getline(ss, field, ','); m.f1 = std::stod(field);
    std::getline(ss, field, ','); m.accuracy = std::stod(field);
    std::getline(ss, field, ','); m.auroc = std::stod(field);
    std::getline(ss, field, ','); m.TP = std::stoi(field);
    std::getline(ss, field, ','); m.FP = std::stoi(field);
    std::getline(ss, field, ','); m.TN = std::stoi(field);
    std::getline(ss, field, ','); m.FN = std::stoi(field);
    
    return m;
}