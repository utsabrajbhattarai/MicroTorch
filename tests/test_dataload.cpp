#include <catch2/catch_test_macros.hpp>
#include "microtorch/data_loader.hpp"

#include <fstream>    //for creating temporary CSV file
#include <cstdio>      // for std::remove()
#include <iostream>
#include <catch2/catch_approx.hpp>

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

TEST_CASE("load_labels maps class values correctly", "[data_loader]") {
    // Create a temporary CSV file with header like that of original dataset
    std::ofstream file("test_labels.csv");

    file << "txID, class\n";
    file << "111,1\n";
    file << "222,2\n";
    file << "333,unknown\n";

    file.close();

    // Loading the labels from the CSV
    std::vector<int> labels = microtorch::load_labels("test_labels.csv", true);

    // Checking the size of the labels vector
    REQUIRE(labels.size() == 3);

    // Checking the mapped label values
    REQUIRE(labels[0] == 1);  // Original class 1 maps to label 1
    REQUIRE(labels[1] == 0);  // Original class 2 maps to label 0
    REQUIRE(labels[2] == -1);  // Original class "unknown" maps to label -1

    // Delete the temporary CSV file
    std::remove("test_labels.csv");

    
}

TEST_CASE("build_txid_index and load_edges work correctly", "[data_loader]")
{
    // Create a temporary features-style CSV file (txId in column 0, no header)
    std::ofstream features_file("test_features.csv");
    features_file << "500,0.1\n";
    features_file << "900,0.7\n";
    features_file << "200,0.3\n";
    features_file << "700,0.9\n";
    features_file.close();

    // Build the txId -> row index map
    std::unordered_map<long long, int> index = microtorch::build_txid_index("test_features.csv");

    // Checking the map maps each txId to the correct row
    REQUIRE(index.at(500) == 0);
    REQUIRE(index.at(900) == 1);
    REQUIRE(index.at(200) == 2);
    REQUIRE(index.at(700) == 3);

    // Create a temporary edge list CSV file (with header)
    std::ofstream edges_file("test_edges.csv");
    edges_file << "txId1,txId2\n";
    edges_file << "900,200\n";
    edges_file << "700,500\n";
    edges_file.close();

    // Load the edges, translated into row-index pairs
    std::vector<std::pair<int,int>> edges = microtorch::load_edges("test_edges.csv", index, true);

    // Checking the size
    REQUIRE(edges.size() == 2);

    // Checking each edge translates to the correct row indices
    REQUIRE(edges[0].first == 1);   // 900 -> row 1
    REQUIRE(edges[0].second == 2);  // 200 -> row 2

    REQUIRE(edges[1].first == 3);   // 700 -> row 3
    REQUIRE(edges[1].second == 0);  // 500 -> row 0

    // Clean up temp files
    std::remove("test_features.csv");
    std::remove("test_edges.csv");
}

TEST_CASE("load_elliptic assembles EllipticData correctly", "[data_loader]")
{
    // writing the features file for test
    std::ofstream features_file("test_elliptic_features.csv");
    features_file << "500,1,0.1,0.2\n";
    features_file << "900,1,0.3,0.4\n";
    features_file << "200,1,0.5,0.6\n";
    features_file << "700,1,0.7,0.8\n";
    features_file.close();

    // writing the classes file
    std::ofstream classes_file("test_elliptic_classes.csv");
    classes_file << "txId,class\n";
    classes_file << "500,2\n";        // licit
    classes_file << "900,1\n";        // illicit
    classes_file << "200,unknown\n";  // unknown
    classes_file << "700,2\n";        // licit
    classes_file.close();

    // writing the tiny edges file for test
    std::ofstream edges_file("test_elliptic_edges.csv");
    edges_file << "txId1,txId2\n";
    edges_file << "900,200\n";
    edges_file << "700,500\n";
    edges_file.close();

    
    microtorch::EllipticData data = microtorch::load_elliptic(
        "test_elliptic_features.csv",
        "test_elliptic_classes.csv",
        "test_elliptic_edges.csv"
    );

    // checking for x matrix
    REQUIRE(data.X.rows() == 4);
    REQUIRE(data.X.cols() == 2);
    REQUIRE(data.X(0, 0) == Catch::Approx(0.1));
    REQUIRE(data.X(0, 1) == Catch::Approx(0.2));
    REQUIRE(data.X(3, 0) == Catch::Approx(0.7));
    REQUIRE(data.X(3, 1) == Catch::Approx(0.8));

    // checking for y
    REQUIRE(data.Y(0, 0) == Catch::Approx(1.0));   // 500 is licit -> [1,0]
    REQUIRE(data.Y(0, 1) == Catch::Approx(0.0));
    REQUIRE(data.Y(1, 0) == Catch::Approx(0.0));   // 900 is illicit -> [0,1]
    REQUIRE(data.Y(1, 1) == Catch::Approx(1.0));
    REQUIRE(data.Y(2, 0) == Catch::Approx(0.0));   // 200 is unknown -> [0,0]
    REQUIRE(data.Y(2, 1) == Catch::Approx(0.0));
    REQUIRE(data.Y(3, 0) == Catch::Approx(1.0));   // 700 is licit -> [1,0]
    REQUIRE(data.Y(3, 1) == Catch::Approx(0.0));

    // in our tiny file 0,1,3 rows are labelled and 2 is not so checking if labelled rows are identified correctly
    REQUIRE(data.labeled_rows.size() == 3);
    REQUIRE(std::find(data.labeled_rows.begin(), data.labeled_rows.end(), 0) != data.labeled_rows.end());
    REQUIRE(std::find(data.labeled_rows.begin(), data.labeled_rows.end(), 1) != data.labeled_rows.end());
    REQUIRE(std::find(data.labeled_rows.begin(), data.labeled_rows.end(), 3) != data.labeled_rows.end());
    REQUIRE(std::find(data.labeled_rows.begin(), data.labeled_rows.end(), 2) == data.labeled_rows.end());

    // checking symmetric edges and self loops
    REQUIRE(data.A.coeff(1, 2) == Catch::Approx(1.0));   // 900-200 edge
    REQUIRE(data.A.coeff(2, 1) == Catch::Approx(1.0));   // symmetric
    REQUIRE(data.A.coeff(3, 0) == Catch::Approx(1.0));   // 700-500 edge
    REQUIRE(data.A.coeff(0, 3) == Catch::Approx(1.0));   // symmetric
    REQUIRE(data.A.coeff(0, 0) == Catch::Approx(1.0));   // self-loop
    REQUIRE(data.A.coeff(1, 1) == Catch::Approx(1.0));   // self-loop
    REQUIRE(data.A.coeff(2, 2) == Catch::Approx(1.0));   // self-loop
    REQUIRE(data.A.coeff(3, 3) == Catch::Approx(1.0));   // self-loop
    REQUIRE(data.A.coeff(0, 1) == Catch::Approx(0.0));   // no edge between 500 and 900

    // cleaning up all files
    std::remove("test_elliptic_features.csv");
    std::remove("test_elliptic_classes.csv");
    std::remove("test_elliptic_edges.csv");
}

