================================================================================
  MicroTorch — A From-Scratch C++ Automatic Differentiation Engine
  Object-Oriented Programming (ENCT 151)
  Utsab Raj Bhattarai · Sonik Bhusal · Shubham Pokhrel
================================================================================


--------------------------------------------------------------------------------
1. REQUIRED LIBRARIES
--------------------------------------------------------------------------------

The engine itself depends ONLY on Eigen (header-only, for matrix arithmetic).
The graphical viewers additionally use raylib. Both libraries are fetched
AUTOMATICALLY by CMake at configure time — you do NOT need to install them
manually. An internet connection is required the first time you configure the
project so these can be downloaded.

  - Eigen 3.4.0        (matrix arithmetic; used by the engine)
  - raylib 5.5         (graphics; used only by the two GUI viewers)
  - Catch2 3.5.4       (build-time test harness only; NOT part of the engine)

For reference, the library sources are:
  Eigen   : https://gitlab.com/libeigen/eigen
  raylib  : https://github.com/raysan5/raylib
  Catch2  : https://github.com/catchorg/Catch2


--------------------------------------------------------------------------------
2. PREREQUISITES (tools you must have installed)
--------------------------------------------------------------------------------

  - A C++17 compiler   (g++ / clang++ / MSVC)
  - CMake 3.16 or newer
  - Git                (CMake uses it to fetch the libraries above)
  - An internet connection on the first configure


--------------------------------------------------------------------------------
3. HOW TO BUILD
--------------------------------------------------------------------------------

From the project root directory (the folder containing the top-level
CMakeLists.txt), run:

    cmake -B build -S .
    cmake --build build

The first command configures the project and downloads Eigen, raylib, and
Catch2. The second compiles everything: the engine library, the two training
programs, the two GUI viewers, and the test suite.

  Note: the first build takes a few minutes because the libraries are being
  downloaded and compiled. Subsequent builds are much faster.

For a faster (optimized) build, configure in Release mode:

    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
    cmake --build build


--------------------------------------------------------------------------------
4. HOW TO RUN
--------------------------------------------------------------------------------

All executables are produced inside the "build" directory. Run them from the
PROJECT ROOT (so that data and output files are found/written in the right
place).

  (a) Train and evaluate the Graph Neural Network (AML on Elliptic dataset):

        ./build/app/gnn_train

  (b) Train the diffusion model and generate animation frames:

        ./build/app/diffusion_train

  (c) Launch the GNN analyst dashboard (reads gui_artifacts/*.csv):

        ./build/app/gui/gnn_gui

  (d) Launch the diffusion viewer (reads frames/*.csv):

        ./build/app/gui/diffusion_viewer

  (On Windows the executables end in .exe, e.g. build\app\gnn_train.exe)


--------------------------------------------------------------------------------
5. HOW TO RUN THE TESTS (optional)
--------------------------------------------------------------------------------

Each engine operation is verified by a gradient check (analytic gradients
compared against numerical finite differences). To run the full test suite:

    cd build
    ctest

Each operation is registered as its own test, so a failure names the exact
component that broke.


--------------------------------------------------------------------------------
6. NOTES
--------------------------------------------------------------------------------

  - The Elliptic Bitcoin dataset is expected in the data/ directory. It is not
    redistributed here due to size; see data/README.md for the source.

  - The GUI viewers read pre-generated CSV artifacts that are included with the
    submission (gui_artifacts/ and frames/), so they can be run directly
    without retraining.

================================================================================