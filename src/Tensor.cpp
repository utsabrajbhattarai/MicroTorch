
#include "microtorch/Tensor.hpp"

#include <functional>
#include <unordered_set>
#include <vector>


namespace microtorch {

void Tensor::zero_grad() {
    grad.setZero();   //sets gradient to zero, for start of new epoch
}

void Tensor::backward() {
    std::vector<Tensor*> topo;              //lists tensors in topological order (when back traversing needs to be reversed)
    std::unordered_set<Tensor*> visited;    //set of the tensors* which are already visited to reduce/remove redundancy

    //first we need to declare a lambda function then call it
    //the [&] is for capturing its own name so when we are doing build_topo it understands what function we are calling also gets topo, visited.
    std::function<void(Tensor*)> build_topo = [&](Tensor* node) {
        if (visited.count(node)) return;    //if already visited no need to compute grad again
        visited.insert(node);
        for (const TensorPtr& parent : node->parents) {
            build_topo(parent.get());       //basically creates a raw pointer of same sharedptr (no free /delete required just a temporary ptr)
        }
        topo.push_back(node);               //first push the bottom most layer or the earliest or most trivial parents then at the top of graph
    };

    build_topo(this);                       //way of calling the above function by the object we are working with atm

    grad.setOnes();                         //sets d(output)/output = 1 otherwise would give us 0 for all the graph

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {        //we need to begin from reverse cause while backward the top most layer comes first and topo is just opposite
        Tensor* node = *it;
        if (node->backward_fnc) {           //if no backward_function leaves it empty 
            node->backward_fnc();
        }
    }
}

}  // namespace microtorch