# app/: thin programs that link against libmicrotorch. If these stay short
# and readable, that's evidence the engine -- not glue code -- is doing the
# real work, which is the whole "engine is the product" framing.

add_executable(gnn_train gnn_train.cpp)
target_link_libraries(gnn_train PRIVATE microtorch)

add_executable(diffusion_train diffusion_train.cpp)
target_link_libraries(diffusion_train PRIVATE microtorch)
