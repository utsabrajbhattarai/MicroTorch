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