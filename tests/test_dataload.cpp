#include <catch2/catch_test_macros.hpp>
#include "microtorch/data_loader.hpp"

#include <fstream>    //for creating temporary CSV file
#include <cstdio>      // for std::remove()

TEST_CASE("CSV loader loads matrix correctly", "[data_loader]")
{
    // Create a temporary CSV file
    std::ofstream file("test.csv");

    file << "1.5,2.5\n";
    file << "3.5,4.5\n";
    file << "5.5,6.5\n";

    file.close();

    // Loading the CSV into an Eigen matrix
    Eigen::MatrixXd matrix = microtorch::load_csv("test.csv");

    // Checking matrix dimensions
    REQUIRE(matrix.rows() == 3);
    REQUIRE(matrix.cols() == 2);

    // Checking matrix values
    REQUIRE(matrix(0,0) == 1.5);
    REQUIRE(matrix(0,1) == 2.5);

    REQUIRE(matrix(1,0) == 3.5);
    REQUIRE(matrix(1,1) == 4.5);

    REQUIRE(matrix(2,0) == 5.5);
    REQUIRE(matrix(2,1) == 6.5);

    // Delete the temporary CSV file
    std::remove("test.csv");
}