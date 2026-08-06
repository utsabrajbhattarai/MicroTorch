# libmicrotorch: the engine, built once, linked everywhere else.
#
# Add new .cpp files here as ops/optimizers get built. Each one should have
# a matching test file in tests/ before it's considered "done" -- the
# gradient-check harness lives in tests/, called once per op.

add_library(microtorch STATIC
    # ---- core ----
    Tensor.cpp
    gradient_check.cpp #for checking the gradients of the ops, used in tests


    # ---- data ----
    data/toy_data.cpp       #for diffusion model, toy data generation
    data/data_loader.cpp    #for GNN model data loading from csv

    # ---- ops/ (one file per op, add as you build them) ----
    ops/matmul.cpp
    ops/broadcast_add.cpp
    ops/add.cpp
    ops/relu.cpp
    ops/neighbor_aggregation.cpp
    ops/reduction.cpp          # sum and mean
    ops/mse_loss.cpp
    ops/softmax_cross_entropy.cpp   #softmax and cross-entropy

    # ---- optim/ ----
    optim/sgd.cpp
    optim/adam.cpp
    
    # ---- logger/ ----
    logger.cpp 

    # ---- gnn/ : node-classification model on top of the ops above ----
    gnn/gnn_model.cpp

    # ---- diffusion/ : MLP denoiser + noise schedule, stretch goal ----
    diffusion/diffusion_model.cpp
    diffusion/noise_schedule.cpp
)

target_include_directories(microtorch
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(microtorch
    PUBLIC Eigen3::Eigen
)

if(MSVC)
    target_compile_options(microtorch PRIVATE /W4)
else()
    target_compile_options(microtorch PRIVATE -Wall -Wextra)
endif()