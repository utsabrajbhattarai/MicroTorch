# MicroTorch

A from-scratch C++ automatic-differentiation engine with built-in gradient
verification, demonstrated by implementing a graph neural network for
anti-money-laundering detection and a diffusion-based generative model --
no ML libraries used.

The engine is the product. The GNN and diffusion model are demos that
prove it's a genuine framework, not a one-architecture hack -- both are
built from the same minimal op set with no engine-side hacks for either.

## Status

Proposal/planning stage. See `docs/` for the methodology and week-by-week
plan. No engine code written yet -- this repo currently holds the build
skeleton only.

## Build

Requires CMake 3.16+ and a C++17 compiler. Eigen and Catch2 are pulled
automatically via CMake's `FetchContent` -- no manual install needed.

```bash
cmake -B build -S .
cmake --build build
ctest --test-dir build
```

`ctest` runs the full test suite, including the gradient-check harness for
every op. A failing op shows up as a named test failure, not a silent bug.

## Layout

```
include/microtorch/   public headers (Tensor, op declarations)
src/                  engine implementation, builds as libmicrotorch
  ops/                matmul, broadcast_add, relu, neighbor_aggregation, ...
  optim/              sgd, adam
  gnn/                node-classification model (must-ship)
  diffusion/          MLP denoiser + noise schedule (stretch goal)
tests/                Catch2, mirrors src/ 1:1 -- every op has a test file
app/                  thin entry points: gnn_train, diffusion_train
data/                 gitignored -- see data/README.md for sourcing
docs/                 methodology, week plan, proposal
```

## Datasets

- **GNN**: [Elliptic Bitcoin dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
  -- node classification, sparse per-timestep batching. See `data/README.md`.
- **Diffusion**: 2D toy data (spiral / two-moons), generated locally, not
  downloaded. Separate from the GNN dataset by design -- diffusion and
  classification are different tasks and don't need to share data.

## Team

Two teammates own bounded, independently-testable leaf tasks (data
pipeline, SGD/Adam optimizers, eval metrics, toy-data generation, loss
logging) -- see `docs/` for the full split. The engine core, the GNN
forward/backward logic, and the diffusion training loop are owned
end-to-end by the project lead.
