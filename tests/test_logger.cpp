#include <catch2/catch_test_macros.hpp> //for the Catch2 testing framework
#include "microtorch/logger.hpp" //header file for logger
#include <fstream> //for file operations
#include <sstream> //for string stream operations

using namespace microtorch;

TEST_CASE("log_training_csv writes correct header and rows", "[logger]") { //test case to verify that the log_training_csv function writes the correct header and rows to a CSV file
    std::vector<double> dummy_losses = {2.5, 2.1, 1.8}; //dummy loss values for testing
    std::string path = "test_output.csv"; //path to the output CSV file for testing

    log_training_csv(path, dummy_losses); //call the log_training_csv function to write the dummy loss values to the specified CSV file

    std::ifstream file(path); //open the output CSV file for reading
    REQUIRE(file.is_open()); //check that the file was opened successfully

    std::string line; //variable to hold each line read from the CSV file
    std::getline(file, line); //read the first line (header) from the CSV file
    REQUIRE(line == "epoch,loss"); //check that the header line matches the expected header 

    std::getline(file, line); //read the second line/first data row from the CSV file
    REQUIRE(line == "0,2.5"); //check that the first data row matches the expected values

    std::getline(file, line); //read the third line/second data row from the CSV file
    REQUIRE(line == "1,2.1"); //check that the second data row matches the expected values

    std::getline(file, line); //read the fourth line/third data row from the CSV file
    REQUIRE(line == "2,1.8"); //check that the third data row matches the expected values

    file.close(); //close file
}