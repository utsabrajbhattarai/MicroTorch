# MicroTorch

**A from-scratch reverse-mode automatic differentiation engine in C++ — with a Graph Neural Network and a Diffusion model built on top to prove it's general.**

<p align="center">
  <img alt="language" src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white">
  <img alt="build" src="https://img.shields.io/badge/build-CMake-064F8C?logo=cmake&logoColor=white">
  <img alt="deps" src="https://img.shields.io/badge/engine%20deps-Eigen%20only-2F7E78">
  <img alt="tests" src="https://img.shields.io/badge/gradients-finite--diff%20checked-E8743B">
  <img alt="course" src="https://img.shields.io/badge/course-OOP%20(ENCT%20151)-8A8175">
</p>

---

## What is this?

Every deep-learning framework — PyTorch, TensorFlow — is built around one hidden mechanism: **automatic differentiation**. MicroTorch reconstructs that mechanism from first principles in C++, depending only on [Eigen](https://eigen.tuxfamily.org) for matrix arithmetic and **no machine-learning frameworks**.

The engine isn't the demo — **the engine _is_ the product.** To prove it's a genuinely general framework rather than something tailored to one model, we built two fundamentally different models on the exact same operations, tensors, optimizers, and backward pass:

| Model | Type | Task |
|---|---|---|
| **Graph Neural Network** | Discriminative | Anti-money-laundering node classification on the Elliptic Bitcoin dataset |
| **Diffusion Model** | Generative | Generating 2-D shapes from pure noise (DDPM) |

Two interactive [raylib](https://www.raylib.com) desktop viewers make the results tangible: an AML analyst dashboard and an animated denoising viewer.

---

## Highlights

-  **Hand-built autograd** — a `Tensor` computation graph with reverse-mode backpropagation via topological traversal.
-  **Every gradient verified** — each operation is checked against a numerical finite-difference estimate (the incorruptible ground truth), catching both implementation and derivation bugs.
-  **Graph Neural Network** — sparse neighbor aggregation, semi-supervised masked loss, and a controlled GNN-vs-MLP ablation.
-  **Diffusion model** — linear noise schedule, DDPM ancestral sampling, and a sinusoidal timestep embedding (Transformer-style).
-  **Two GUIs** — decoupled from the engine through CSV artifacts; they read files, they don't link the engine.
-  **OOP by principle** — encapsulation where invariants exist, an abstract `Optimizer` hierarchy, three flavors of polymorphism, and composition as the structural backbone.

---

## Architecture

Three layers, strictly one-directional dependencies. The engine depends only on Eigen; the demos depend on the engine; the GUIs depend on neither — they're decoupled through the filesystem.

```
┌──────────────────────────────────────────────────────────────────┐
│  ENGINE  (libmicrotorch, static library)          depends: Eigen  │
│  Tensor · reverse-mode backward · ops · optimizers · grad-check   │
└──────────────────────────────────────────────────────────────────┘
            │                                    ▲
            ▼                                    │  (reads CSV artifacts)
┌───────────────────────────┐        ┌───────────────────────────────┐
│  DEMOS  (apps)            │        │  GUIs  (raylib)                │
│  gnn_train                │──CSV──▶│  gnn_dashboard                 │
│  diffusion_train          │──CSV──▶│  diffusion_viewer              │
└───────────────────────────┘        └───────────────────────────────┘
```

---

## Repository layout

```
MicroTorch/
├── include/microtorch/     # public headers (the API surface)
│   ├── Tensor.hpp          #   the core autograd node
│   ├── ops/                #   one header per operation
│   ├── optim/              #   Optimizer (abstract) → SGD, Adam
│   ├── gnn/ · diffusion/   #   the two models
│   └── data/ · eval/ · gui/
├── src/                    # implementations → compiled into libmicrotorch
├── app/
│   ├── gnn_train.cpp       #   trains + evaluates the GNN (+ ablation)
│   ├── diffusion_train.cpp #   trains the denoiser, exports animation frames
│   └── gui/                #   gnn_dashboard + diffusion_viewer (raylib)
├── tests/                  # per-operation gradient checks
├── gui_artifacts/          # CSVs the dashboard reads
├── frames/                 # per-shape diffusion animation CSVs
└── data/                   # Elliptic dataset (see data/README.md)
```

---

## Getting started

### Prerequisites

- A **C++17** compiler (g++ / clang++ / MSVC)
- **CMake** ≥ 3.16
- **Git** and an internet connection on the first configure

> Eigen, raylib, and Catch2 are fetched **automatically** by CMake at configure time — nothing to install by hand.

### Build

```bash
cmake -B build -S .        # configures + downloads dependencies (first run takes a few minutes)
cmake --build build        # compiles the engine, demos, GUIs, and tests
```

For fast training runs, build optimized:

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Run

Run from the project root so data and output files resolve correctly.

```bash
./build/app/gnn_train              # train + evaluate the GNN, export dashboard artifacts
./build/app/diffusion_train        # train the diffusion model, export animation frames
./build/app/gui/gnn_gui            # launch the AML analyst dashboard
./build/app/gui/diffusion_viewer   # launch the animated denoising viewer
```

*(On Windows, executables end in `.exe`.)*

---

## The two demos

## Graph Neural Network — AML on Elliptic Bitcoin

A two-layer graph convolutional network that classifies Bitcoin transactions as licit or illicit. Each node learns from its transaction neighbors via sparse neighbor aggregation. The **dashboard** ranks accounts by a continuous risk score, shows evaluation metrics, and draws an interactive transaction ego-graph with directional arrows.

> **Honest finding.** In a controlled ablation, disabling neighbor aggregation (reducing the model to a plain MLP) *matched or beat* the GNN — the Elliptic features already encode neighborhood information, so extra aggregation over-smooths. Reported as a real result, not hidden. The value is that the engine can express and rigorously compare both.

## Diffusion Model — shapes from noise

A small MLP denoiser trained to reverse a gradual noising process. Starting from pure Gaussian noise, DDPM ancestral sampling iteratively removes predicted noise until a shape emerges. A **sinusoidal timestep embedding** (the Transformer positional-encoding trick) was the key unlock that broke a training plateau. The **viewer** animates the denoising with a scrubbable timeline and a shape gallery.

---

## How the engine works

A `Tensor` is a node in a computation graph. It holds its value, its gradient, its parents, and a closure describing how to differentiate the operation that produced it:

```cpp
class Tensor {
    Eigen::MatrixXd data;                 // the value
    Eigen::MatrixXd grad;                 // accumulated gradient
    std::vector<TensorPtr> parents;       // where it came from
    std::function<void()> backward_fnc;   // how to push gradient to parents
};
```

The forward pass builds the graph but computes no gradients. Calling `backward()` on a scalar loss performs a topological sort, seeds the loss gradient to 1, and walks the graph **in reverse**, firing each node's `backward_fnc` so gradient flows from the loss back to every parameter via the chain rule.

**The backward pass is data — a list of closures — not control flow.** Each operation records its own derivative rule as a captured lambda, giving runtime polymorphism through type erasure without a parallel class hierarchy.

---

## Testing

Every operation's backward pass is verified by gradient checking — comparing the analytic gradient against a central finite-difference estimate. Because finite differences depend only on the definition of a derivative, they catch wrong *math*, not just wrong *code*.

```bash
cd build && ctest        # each operation is a separate test; a failure names the exact component
```

> Catch2 is a **build-time test harness only** — it is not linked into or part of the engine, which depends solely on Eigen.

---

## Tech stack

| | |
|---|---|
| **Language** | C++17 |
| **Engine dependency** | Eigen 3.4 (matrix arithmetic) |
| **GUI** | raylib 5.5 |
| **Build** | CMake (FetchContent for all deps) |
| **Testing** | Catch2 (build-time only) |

---

## Authors

Built for the Object-Oriented Programming course (ENCT 151), Institute of Engineering, Pulchowk Campus, Tribhuvan University.

- **Utsab Raj Bhattarai**
- **Sonik Bhusal**
- **Shubham Pokhrel**

Developed collaboratively with Git throughout — incremental commits, feature branches per component, and reviews before merging.

---

## License

Academic project. See repository for details.
