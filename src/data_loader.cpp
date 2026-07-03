#include "microtorch/data_loader.hpp"

#include <fstream>     // for file I/O
#include <sstream>     // for string splitting
#include <vector>      // stores values temporarily
#include <stdexcept>

using namespace std;

namespace microtorch {

Eigen::MatrixXd load_csv(const string& path)
{
    ifstream file(path);  // Open the file

    if (!file.is_open())
    {
        throw runtime_error("Could not open file");
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

}



