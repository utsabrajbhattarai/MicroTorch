#pragma once
#include <vector>
#include <string>

namespace microtorch {

void log_training_csv(const std::string& filepath, const std::vector<double>& losses);
//function to log the training loss to a CSV file at the specified filepath.
//Each entry in the losses vector corresponds to a loss value for a training iteration.

}