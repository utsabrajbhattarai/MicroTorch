#include "microtorch/logger.hpp" //header file for logger
#include <fstream> //for file operations

namespace microtorch {

void log_training_csv(const std::string& filepath, const std::vector<double>& losses) { 
    std::ofstream file(filepath); //open the file at the specified filepath for writing

    file << "epoch,loss\n";  //header for the CSV file

    for (size_t i = 0; i < losses.size(); ++i) {
        file << i << "," << losses[i] << "\n"; //write each epoch and its corresponding loss value to the CSV file
    }

    file.close();
}

}