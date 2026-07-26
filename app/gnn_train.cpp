//loads the elliptic dataset, builds the GNN model from engine ops, trains with Adam, evaluates against the published ~95% AUROC baseline, and runs the MLP-vs-GNN ablationby toggling neighbor_aggregation off.

#include "microtorch/Tensor.hpp"
#include "microtorch/gnn/gnn_model.hpp"
#include "microtorch/ops.hpp"
#include "microtorch/data/data_loader.hpp"
#include "microtorch/optim/adam.hpp"
#include <algorithm>
#include <iostream>
#include <random>

const int EPOCH_N = 100;  

using namespace microtorch;

int main() {
    
    //1.0.0)load the data:
    EllipticData data = load_elliptic(
    "data/elliptic_bitcoin_dataset/elliptic_txs_features.csv",  //features file
    "data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv",   //labels file
    "data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"   //edgelist file
    );
    int N = data.X.rows();  //no. of rows
    int F = data.X.cols();  //no. of feats
    TensorPtr X = make_tensor(data.X);

    //1.0.1) Startified Split (due to class imbalance)
    std::vector<int> illicit_rows, licit_rows; //storing licit and illicit rows independently
    for(int r : data.labeled_rows) {
        if (data.Y(r,1) == 1.0){
            illicit_rows.push_back(r);
        }
        else{
            licit_rows.push_back(r);
        }
    }
    std::mt19937 generator(95); //a seeded generator 
    std::shuffle(illicit_rows.begin(), illicit_rows.end(), generator); //shuffling illicit rows
    std::shuffle(licit_rows.begin(), licit_rows.end(), generator); //shuffling licit rows

    std::vector<int> train_rows, test_rows; //test rows and test rows 80/20 split

    int illicit_train = static_cast<int>(0.8 * illicit_rows.size()); //size of illicit train
    train_rows.insert(train_rows.end(), illicit_rows.begin(), illicit_rows.begin() + illicit_train); //inserting illicit train rows
    test_rows.insert (test_rows.end(),  illicit_rows.begin() + illicit_train, illicit_rows.end()); //inserting illicit test rows

    int licit_train = static_cast<int>(0.8 * licit_rows.size()); //size of licit train 
    train_rows.insert(train_rows.end(), licit_rows.begin(), licit_rows.begin() + licit_train); 
    test_rows.insert (test_rows.end(),  licit_rows.begin() + licit_train, licit_rows.end());

    //sanity check for checking a balanced split:
    // int illicit_test = illicit_rows.size() - illicit_train;
    // std::cout << "train: " << train_rows.size() << " nodes, " << illicit_train << " illicit ("
    //           << (100.0 * illicit_train / train_rows.size()) << "%)\n";
    // std::cout << "test:  " << test_rows.size() << " nodes, " << illicit_test << " illicit ("
    //           << (100.0 * illicit_test / test_rows.size()) << "%)\n";

    //2) Masking labelled and unlabelled
    Eigen::MatrixXd train_mask = Eigen::MatrixXd::Zero(N, 1); //creating masked mapping for labelled and unlablled rows of train set
    for (int r : train_rows) train_mask(r, 0) = 1.0;    //train_mask only trains the masked row 
    double num_train = train_rows.size();

    Eigen::MatrixXd test_mask = Eigen::MatrixXd::Zero(N, 1); //creating masked mapping for labelled and unlablled rows of test set
    for (int r : test_rows) test_mask(r, 0) = 1.0;  //so to calculate loss only on test nodes
    double num_test = test_rows.size();


    //3) Model and Optimizer
    GNNModel model(F, 128, 2);
    Adam opt(model.parameters(),0.01);  

    for (int epoch = 0; epoch<=EPOCH_N; epoch++){
        
        // forward through all of train set, loss on train mask, backward, step(update gradient)
        TensorPtr logits = model.forward(data.A, X); //creating logits
        TensorPtr train_loss   = softmax_cross_entropy(logits, data.Y, train_mask, num_train); //finding the loss
        opt.zero_grad();    //reseting the previous grads to zero cause there's a += in each grad which may cause carry effect
        train_loss->backward();   //backwarding through the graph
        opt.step();         //updating parameter/ a step in optimization


        //evaluate test loss from same logits with no backward or step
        if (epoch % 10 == 0) {
        TensorPtr test_loss = softmax_cross_entropy(logits, data.Y, test_mask, num_test);
        std::cout << "epoch " << epoch
                  << "  train_loss " << train_loss->data(0,0)
                  << "  test_loss "  << test_loss->data(0,0) << std::endl;
    }


    }

    return 0;
}
