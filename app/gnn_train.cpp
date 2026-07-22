//loads the elliptic dataset, builds the GNN model from engine ops, trains with Adam, evaluates against the published ~95% AUROC baseline, and runs the MLP-vs-GNN ablationby toggling neighbor_aggregation off.

#include "microtorch/Tensor.hpp"
#include "microtorch/gnn/gnn_model.hpp"
#include "microtorch/ops.hpp"
#include "microtorch/data/data_loader.hpp"
#include "microtorch/optim/adam.hpp"
#include <iostream>

using namespace microtorch;

int main() {
    
    //1)load the data:
    EllipticData data = load_elliptic(
    "data/elliptic_bitcoin_dataset/elliptic_txs_features.csv",  //features file
    "data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv",   //labels file
    "data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"   //edgelist file
    );
    int N = data.X.rows();
    int F = data.X.cols();
    TensorPtr X = make_tensor(data.X);

    return 0;
}
