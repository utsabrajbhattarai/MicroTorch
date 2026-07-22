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
    int N = data.X.rows();  //no. of rows
    int F = data.X.cols();  //no. of feats
    TensorPtr X = make_tensor(data.X);


    //2) Masking labelled and unlabelled
    Eigen::MatrixXd mask = Eigen::MatrixXd::Zero(N,1);  //creating masked mapping for labelled and unlablled rows
    for (int r: data.labeled_rows) mask(r,0) = 1.0;     //looping through labelled rows and changing their mask to 1
    double num_labeled = data.labeled_rows.size();      //number of labelled rows


    //3) Model and Optimizer
    GNNModel model(F, 128, 2);
    Adam opt(model.parameters(),0.01);  

    return 0;
}
