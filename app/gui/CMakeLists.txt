include(FetchContent)
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG        5.5
)
FetchContent_MakeAvailable(raylib)

add_executable(gnn_gui main.cpp csv_loader.cpp)
target_link_libraries(gnn_gui PRIVATE raylib) 

add_executable(diffusion_viewer diffusion_viewer.cpp)
target_link_libraries(diffusion_viewer PRIVATE raylib)
