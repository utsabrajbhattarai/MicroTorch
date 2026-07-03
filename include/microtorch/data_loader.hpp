#pragma once
#include <Eigen/Dense>  //for Eigen::MatrixXd
#include <string>     // for std::string

// created microtorch namespace including load_csv function 
namespace microtorch{
    Eigen::MatrixXd load_csv(const std::string& path);
}