#include "microtorch/data/data_loader.hpp"

#include <fstream>     // for file I/O
#include <sstream>     // for string splitting
#include <vector>      // stores values temporarily
#include <stdexcept>
#include <unordered_map> // for build_txid_index function
#include <set>

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

        auto it1 = txid_to_row.find(txid1); //returns an interator to entry or .end() if not found
        auto it2 = txid_to_row.find(txid2);
        if (it1 == txid_to_row.end() || it2 == txid_to_row.end()) {
            continue;   //one endpoint isn't a known node, skip this edge
        }
        int row1 = it1->second; //if none edge is missing pulling the row index
        int row2 = it2->second;


        // 4. push_back a pair {row1, row2} into edges
        edges.push_back({row1, row2});
    }

    return edges;
}

EllipticData load_elliptic(const string& features_path, const string& classes_path, const string& edges_path)
{
    EllipticData data; // defining data as struct variable for EllipticData
    unordered_map<long long, int> txid_to_row = build_txid_index(features_path, false);
    // building index as 0,1.. for row 0, 1 transaction ids in features file

    Eigen::MatrixXd raw_X = load_csv(features_path, false);  // raw_X variable stores matrix from csv features file

    int N = raw_X.rows();

    if (N == 0) {
        throw runtime_error("Loaded features matrix is empty!");  // if 0 rows, file is empty
    }

    data.X = raw_X.rightCols(raw_X.cols() - 2);  // x member variable(eigenmatrixxd) inside struct data now holds (N, 166) matrix. rightCols(m) takes m cols from right side. as we dont require first two cols so, now x has third to last cols in each row of features file
    
    data.Y = Eigen::MatrixXd::Zero(N, 2);  // y member variable(eigenmatrix) has n rows and two cols, 0th col for licit, 1st col for illicit
    data.labeled_rows.clear();

    ifstream class_file(classes_path);
    if (!class_file.is_open()) {
        throw runtime_error("Could not open classes file: ");
    }
    
    // bypass the header of csv file
    string class_header;
    getline(class_file, class_header);

    string line;
    while (getline(class_file, line)) {
        stringstream ss(line);
        string txid_text, class_text; // first value in csv file for classes is transaction id and second is the class 1,2,unknown for illicit, licit, unknown. store transaction id in txid_text and class in class_text
        
        getline(ss, txid_text, ',');
        getline(ss, class_text, ',');
        
        long long txid = stoll(txid_text); //converting string to long long as txid is long long
        
        
        auto it = txid_to_row.find(txid); // this line returns pointer and txid_to_row is unordered map which may return any datatype so using auto
        if (it != txid_to_row.end()) {
            int row_idx = it->second;  // check if txid is found, if found then get its corresponding row id in row_idx.  it->second is done coz in our unordered map the key(first) is transaction id and value(second) is its corresponding row id
            
            if (class_text == "1") {       // Illicit / Fraudulent
                data.Y(row_idx, 1) = 1.0;  // y matrix has two cols, col0 for licit , 1 for illicit, so if transaction = illicit set second column value for corresponding row index to 1
                data.labeled_rows.push_back(row_idx); // labelled_rows storing only rows that are labelled (no unknowns, only licit or illict)
            } 
            else if (class_text == "2") {  // Licit / Legitimate
                data.Y(row_idx, 0) = 1.0;  // 0th column's (licit column) value is 1
                data.labeled_rows.push_back(row_idx);
            }
            // "unknown" tags remain [0, 0] and are skipped from labeled_rows completely
        }
    }
    class_file.close();

    vector<pair<int, int>> mapped_edges = load_edges(edges_path, txid_to_row, true);

    //  Build Sparse Matrix A with Self-Loops and Symmetric Edges
    vector<Eigen::Triplet<double>> triplets;
    triplets.reserve((mapped_edges.size() * 2) + N);  // reserving memory for triplets, *2 is done for both way transactions, and +N for each self loop transaction for n nodes

    set<pair<int, int>> unique_edges; // set is used to declare the pairs coz it avoids duplicates. if somehow in our csv file, same edges are repeated twice(or some nodes have multiple transactions betn them) then this wont add the value(third value of triplet). it would remain as 1.(not 2 if they have two repitions,or 3 for 3 repitions....)

    for (const auto& edge : mapped_edges)
    {
        int u = edge.first;  
        int v = edge.second;
        unique_edges.insert({u, v});  // insert for a edge(u node to v transaction)
        unique_edges.insert({v, u});  // both way transaction (if u to v then v to u)
    }

    for (int i = 0; i < N; ++i)
    {
        unique_edges.insert({i, i});  // for self loop
    }

    for (const auto& e : unique_edges)
    {
        triplets.push_back(Eigen::Triplet<double>(e.first, e.second, 1.0));  // now storing in actual triplets that we want. unique_edges thing was done just for no duplication. so, if there is edge(transaction betn node a and node b then set its value to 1.0 otherwise 0)
    }

    data.A.resize(N, N);
    data.A.setFromTriplets(triplets.begin(), triplets.end()); // setting the adjacency matrix denoted by A according to triplets u just made
    data.A.makeCompressed();  // ignore all 0 values(if a node hasnot transacted with other node then dont store in the sparse matrix)

    //SYMMETRIC NORMALIZATION:
    // D^(-1/2) * A * D^(-1/2)
    //without normalization features sum up so a node with lots of neighbours features explode and cause a huge error:
    Eigen::VectorXd deg = data.A * Eigen::VectorXd::Ones(N);   //row sums = degrees (how many connection a node has including its self loop)

    Eigen::VectorXd dinv(N);
    for (int i = 0; i < N; ++i){
        dinv(i) = (deg(i) > 0.0) ? 1.0 / std::sqrt(deg(i)) : 0.0;   //guard divide by 0 if node is isolated(although we have self loop/node just a prevention)
    }

    data.A = dinv.asDiagonal() * data.A * dinv.asDiagonal();    //the normalization part as a diagonal vector for efficient computing
    data.A.makeCompressed();


    return data;


}
}



