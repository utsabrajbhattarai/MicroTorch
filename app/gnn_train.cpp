//loads the elliptic dataset, builds the GNN model from engine ops, trains with Adam, evaluates against the published ~95% AUROC baseline, and (week 4) runs the MLP-vs-GNN ablationby toggling neighbor_aggregation off.

#include "microtorch/Tensor.hpp"
#include "microtorch/gnn/gnn_model.hpp"
#include "microtorch/ops.hpp"
#include "microtroch/adam.hpp"
#include <iostream> 

int main() {
    // TODO: load Elliptic data, build model, train, evaluate, ablate.
    return 0;
}
