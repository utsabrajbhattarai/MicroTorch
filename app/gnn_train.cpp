//loads the elliptic dataset, builds the GNN model from engine ops, trains with Adam, evaluates against the published ~95% AUROC baseline, and runs the MLP-vs-GNN ablationby toggling neighbor_aggregation off.

#include "microtorch/Tensor.hpp"
#include "microtorch/gnn/gnn_model.hpp"
#include "microtorch/ops.hpp"
#include "microtorch/data/data_loader.hpp"
#include "microtorch/optim/adam.hpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <string>

const int EPOCH_N = 70;  

using namespace microtorch;

double compute_auroc(const std::vector<double>& scores, const std::vector<bool>& is_illicit); //function that returns AUROC score


//runs one seed: split with this seed to train GNN and MLP (for the purpose of analyzing variation of loss with choice of seed (hyperparameter tuning))
void run_seed(unsigned seed, const EllipticData& data, const TensorPtr& X, int N, int F, double& gnn_auroc, double& mlp_auroc);

//runs one full train + evaluate cycle || use_graph = true is the GNN || use_graph = false disables neighbor_aggregation, making it a plain MLP.
//everything else (split, masks, epochs, init) is identical between the two,
//so the metric difference is purely the contribution of the graph structure ie the boolean value of use_graph whose change can be seen in the forward function
double run_experiment(bool use_graph, const EllipticData& data, const TensorPtr& X, const Eigen::MatrixXd& train_mask, double num_train,
                    const Eigen::MatrixXd& test_mask,  double num_test, const std::vector<int>& test_rows, int F);

int main() {

//*************************************************************************************// 
    //0)load the data:
    EllipticData data = load_elliptic(
    "data/elliptic_bitcoin_dataset/elliptic_txs_features.csv",  //features file
    "data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv",   //labels file
    "data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"   //edgelist file
    );
    int N = data.X.rows();  //no. of rows
    int F = data.X.cols();  //no. of feats
    TensorPtr X = make_tensor(data.X);


//*************************************************************************************//
    std::vector<unsigned> seeds = {95, 67, 123};
    double gnn_sum = 0, mlp_sum = 0;
    std::vector<double> gnn_all, mlp_all;

    for (unsigned s : seeds) {
        double g, m;
        std::cout << "\n===== SEED " << s << " =====\n";
        run_seed(s, data, X, N, F, g, m);
        gnn_all.push_back(g); mlp_all.push_back(m);
        gnn_sum += g; mlp_sum += m;
    }

    std::cout << "\n===== SUMMARY over " << seeds.size() << " seeds =====\n";
    std::cout << "GNN range: " << *std::min_element(gnn_all.begin(), gnn_all.end())
              << " to " << *std::max_element(gnn_all.begin(), gnn_all.end()) << "\n";
    std::cout << "MLP range: " << *std::min_element(mlp_all.begin(), mlp_all.end())
              << " to " << *std::max_element(mlp_all.begin(), mlp_all.end()) << "\n";

    return 0;
    
}


//STEP 1 THROUGH 2:

void run_seed(unsigned seed, const EllipticData& data, const TensorPtr& X, int N, int F, double& gnn_auroc, double& mlp_auroc){

//1) Startified Split (due to class imbalance)
    std::vector<int> illicit_rows, licit_rows; //storing licit and illicit rows independently
    for(int r : data.labeled_rows) {
        if (data.Y(r,1) == 1.0){
            illicit_rows.push_back(r);
        }
        else{
            licit_rows.push_back(r);
        }
    }
    std::mt19937 generator(seed); //a seeded generator: seed is now parameterized
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



//*************************************************************************************//
    //2) Masking labelled and unlabelled
    Eigen::MatrixXd train_mask = Eigen::MatrixXd::Zero(N, 1); //creating masked mapping for labelled and unlablled rows of train set
    for (int r : train_rows) train_mask(r, 0) = 1.0;    //train_mask only trains the masked row 
    double num_train = train_rows.size();

    Eigen::MatrixXd test_mask = Eigen::MatrixXd::Zero(N, 1); //creating masked mapping for labelled and unlablled rows of test set
    for (int r : test_rows) test_mask(r, 0) = 1.0;  //so to calculate loss only on test nodes
    double num_test = test_rows.size();


     // run both, capturing AUROC (see note below about making run_experiment return it)
    gnn_auroc = run_experiment(true,  data, X, train_mask, num_train, test_mask, num_test, test_rows, F);
    mlp_auroc = run_experiment(false, data, X, train_mask, num_train, test_mask, num_test, test_rows, F);
}





//STEP 3 THROUGH 5:


double run_experiment(bool use_graph, const EllipticData& data, const TensorPtr& X, const Eigen::MatrixXd& train_mask, double num_train,
                    const Eigen::MatrixXd& test_mask,  double num_test, const std::vector<int>& test_rows, int F){


//*************************************************************************************//
    //3) Model and Optimizer
    GNNModel model(F, 128, 2);
    Adam opt(model.parameters(), 0.01);


 //*************************************************************************************//
    //4) Training Loop
    for (int epoch = 0; epoch<=EPOCH_N; epoch++){
        
        // forward through all of train set, loss on train mask, backward, step(update gradient)
        TensorPtr logits = model.forward(data.A, X, use_graph); //creating logits
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

//*************************************************************************************//
    //5) Evaluation Metrics
    int TP = 0, FP = 0, TN = 0, FN = 0; //evaluation matrix
    TensorPtr final_logits = model.forward(data.A, X, use_graph);   //final forward for evaluation purposes

    std::vector<double> scores; //logits2-logits1 scores
    std::vector<bool> is_illicit; //calculates if its illicit or not

    for (int r : test_rows) {
        bool pred_illicit  = final_logits->data(r, 1) > final_logits->data(r, 0);   //predicted positives (softmax is monotonics so logits is good enough)
        bool truth_illicit = (data.Y(r, 1) == 1.0);     //actual positives

        if      ( pred_illicit &&  truth_illicit) TP++;
        else if ( pred_illicit && !truth_illicit) FP++;
        else if (!pred_illicit && !truth_illicit) TN++;
        else                                      FN++;

        //for auroc function:
        scores.push_back(final_logits->data(r, 1) - final_logits->data(r, 0));  //illicit-ness/ score of how illicit it is
        is_illicit.push_back(data.Y(r, 1) == 1.0);  //illicit or not bool
    }
    //metrics used: (conditions are just for sake of guarding division by 0)
    double precision = (TP + FP > 0) ? (double)TP / (TP + FP) : 0.0; //out of all guessed illicit how many were True?
    double recall    = (TP + FN > 0) ? (double)TP / (TP + FN) : 0.0;  //out of all true illicit how many were guessed?
    double f1        = (precision + recall > 0)
                    ? 2 * precision * recall / (precision + recall) : 0.0;  //way to calculate f1 score (harmonic mean of precision and recall)
    double accuracy  = (double)(TP + TN) / test_rows.size();
    double auroc = compute_auroc(scores, is_illicit);   //calling auroc function

    //GNN or MLP?
    std::string model = (use_graph)?"GNN":"MLP";

    //printing the metrices:
    std::cout << "\n\n\n========" << model << "=======\n\n"
    std::cout << "\n=== Test metrics ===\n";
    std::cout << "TP=" << TP << " FP=" << FP << " TN=" << TN << " FN=" << FN << "\n";
    std::cout << "precision " << precision << "  recall " << recall
            << "  F1 " << f1 << "  accuracy " << accuracy << "\n";
    std::cout << "AUROC " << auroc << "\n";


    return auroc;
}








//*************************************************************************************//
//complete auroc score:
double compute_auroc(const std::vector<double>& scores, const std::vector<bool>& is_illicit){
    //pairing each score and is label:
    std::vector<std::pair<double,bool>> score_pair;

    for( size_t i =0; i<scores.size(); i++){
        score_pair.emplace_back(scores[i], is_illicit[i]);
    }

    //using sort lambda function to sort scores based on .first(score)
    std::sort(score_pair.begin(), score_pair.end(),[]
              (const auto& a, const auto &b){
                return a.first < b.first;
              }
    );



    //assigning ranks and summing the illicit ranks:
    int i = 1, n_illicit = 0, n_licit =0 ;
    double R = 0.0;   //a running sum of illicit ranks
    for (const std::pair<double,bool>& x : score_pair){
        if( x.second ){
            R += static_cast<int>(i);  //ranking running sum
            n_illicit++;    //number of illicit case
        }
        else{
            n_licit++;  //numer of licit case
        }
        i++;
    }

    //applying the Mann-Whitney U statistic formula for calculaitng aucroc score:
    double AUROC = (R - static_cast<double>(n_illicit) * (static_cast<double>(n_illicit) + 1) / 2.0) / (static_cast<double>(n_illicit) * n_licit);

    return AUROC;

}


