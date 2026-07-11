#include "microtorch/data_loader.hpp"

#include <fstream>     // for file I/O
#include <sstream>     // for string splitting
#include <vector>      // stores values temporarily
#include <stdexcept>
#include <unordered_map> // for build_txid_index function

using namespace std;

namespace microtorch {

Eigen::MatrixXd load_csv(const string& path, bool has_header)
{
    ifstream file(path);  // Open the file

    if (!file.is_open())
    {
        throw runtime_error("Could not open file");
    }

    if(has_header)
    {
        string header_line;
        getline(file, header_line);  // Read and discard the header line if csv file has a header
    }

    string line;
    vector<vector<double>> data;  // to store the rows as each element of vector in data

    while (getline(file, line))  // read each line from file and store in line
    {
        stringstream ss(line);   // read line part by part (value by value) and store in ss

        vector<double> row;   // creating a vector to store the values of the current row

        string cell;

        while (getline(ss, cell, ','))  //keep reading till u encounter comma and store in cell
        {
            double value = stod(cell); // Convert string to double

            row.push_back(value); // Store value in row
        }
        data.push_back(row); // Store row in data
    }

    if(data.empty()){
        return Eigen::MatrixXd(0, 0); // Returns an empty matrix 
    }

    //Determine matrix size
    int rows = data.size();
    int cols = data[0].size();

    // create matrix of required size
    Eigen::MatrixXd matrix(rows, cols);

    //copy values inside matrix
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            matrix(i, j) = data[i][j];
        }
    }

    return matrix;

}

vector <int> load_labels(const string& path, bool has_header)
{
    ifstream file(path);  // Open the file
    if (!file.is_open())
    {
        throw runtime_error("Could not open file");
    }

    if(has_header)
    {
        string header_line;
        getline(file, header_line);  // Read and discard the header line if csv file has a header
    }

    vector<int> labels;  // to store the labels
    string line;
    

    while(getline(file, line))
    {
        stringstream ss(line); // convert each line to kind of file to read sth at each interval like value by value
        string txid_text, class_text; // to store the two values of each line 

        getline(ss, txid_text, ','); // read till comma and store in txid_text
        getline(ss, class_text, ','); // read the second value of each line and store in class_text

        int label; 

        if (class_text == "1") {
            label = 1;  // if transaction is illicit/fraudulent, label = 1
        }

        else if(class_text == "2"){
            label = 0;  // if transaction is licit/legitimate, label = 2
        }

        else{
            label = -1; // if the transaction type is unknown, its label is assigned -1
        }

        labels.push_back(label); // store the label in the vector labels

    }

    return labels; 
}

unordered_map<long long, int> build_txid_index(const string& path, bool has_header)
{
    ifstream file(path);

    if (!file.is_open())
    {
        throw runtime_error("Could not open file");
    }

    if (has_header)
    {
        string header_line;
        getline(file, header_line);
    }

    unordered_map<long long, int> index;
    string line;
    int row = 0;   //counts the row index starting from 0

    while (getline(file, line))
    {
        stringstream ss(line);
        string txid_text;

        getline(ss, txid_text, ',');   // only take the first value i.e txid

        long long txid = stoll(txid_text);   // convert to a number

        index[txid] = row;  // map the txid to its corresponding row index in the csv file

        row = row + 1;   // move to next row for the next iteration
    }

    return index;
}

vector<pair<int,int>> load_edges(const string& path, const unordered_map<long long, int>& txid_to_row, bool has_header)
{
    ifstream file(path);

    if (!file.is_open())
    {
        throw runtime_error("Could not open file");
    }

    if (has_header)
    {
        string header_line;
        getline(file, header_line);  //bypass the header line if csv file has one
    }

    vector<pair<int,int>> edges;  // pair means storing in (1,2) this format
    string line;

    while (getline(file, line))
    {
        stringstream ss(line);
        string txid1_text, txid2_text;

        getline(ss, txid1_text, ','); // first value of each line is txid1
        getline(ss, txid2_text, ','); // second value of each line is txid2

        long long txid1 = stoll(txid1_text); // convert to number
        long long txid2 = stoll(txid2_text); // convert to number

        int row1 = txid_to_row.at(txid1); // get the row index of txid1 from the unordered_map
        int row2 = txid_to_row.at(txid2); // get the row index of txid2 from the unordered_map

        // 4. push_back a pair {row1, row2} into edges
        edges.push_back({row1, row2});
    }

    return edges;
}
}



