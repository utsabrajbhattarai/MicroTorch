#pragma once
#include "microtorch/Tensor.hpp"
#include <vector>

namespace microtorch {

    // defining sgd class which stores pointer to tensor class and learning rate 
    class SGD{
        public:
        SGD(std::vector<TensorPtr> params, double lr);
        void step();   // each step of updating weight
        void zero_grad();   //setting gradient to zero

        private:
        std::vector<TensorPtr> params_;
        double lr_;
    };

    
}

 
