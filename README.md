# MicroTorch

A from-scratch C++ automatic-differentiation engine with built-in gradient
verification, demonstrated by a graph neural network for anti-money-laundering
detection and a diffusion-based generative model. No machine learning libraries
are used.

The engine is the product. The GNN and the diffusion model are demos that prove
it is a genuine framework rather than a one-architecture hack. Both are built
from the same minimal op set, with no engine-side special-casing for either.

## Status

The core engine is complete and verified. The `Tensor` type and reverse-mode
automatic differentiation work end to end, and the full operation set is
implemented, with every operation checked against finite differences.

Done:
- `Tensor` class and reverse-mode autograd (topological backward pass)
- Operations, each gradient-checked: `matmul`, `broadcast_add`, `relu`, `sum`,
  `mean`, `mse_loss`, `neighbor_aggregation` (sparse), and the fused
  `softmax_cross_entropy`
- A Catch2 test per operation, including a numeric gradient check

In progress or planned:
- Optimizers: SGD and Adam (validated on a toy quadratic)
- Data loading for the Elliptic dataset, plus synthetic 2D shape generators
- GNN assembly (node classification) and the diffusion training loop
- A raylib dashboard for visualization

See `docs/` for the methodology and the week-by-week plan.

## Build

Requires CMake 3.16+ and a C++17 compiler. Eigen and Catch2 are pulled
automatically through CMake `FetchContent`, so no manual install is needed.
raylib is fetched only for the GUI executable, so the engine and its tests do
not depend on it.

```bash
cmake -B build -S .
cmake --build build
ctest --test-dir build
```

`ctest` runs the full suite. Every operation carries a gradient check, so a
wrong backward pass shows up as a named test failure rather than a silent bug
that only surfaces later as a model that never learns.

## Layout

```
include/microtorch/   public headers
  Tensor.hpp          the core tensor type
  ops/                matmul, broadcast_add, relu, reduction, mse_loss,
                      neighbor_aggregation, softmax_cross_entropy
  optim/              sgd, adam
  gnn/                node-classification model header
  diffusion/          denoiser and noise-schedule header
  data/               data_loader (Elliptic) and toy_data (2D shapes)
src/                  engine implementation, builds as libmicrotorch
  ops/ optim/ gnn/ diffusion/ data/   mirror the header tree
tests/                Catch2, mirrors src one-to-one, a test file per op
app/                  entry points: gnn_train, diffusion_train
  gui/                raylib dashboard, links libmicrotorch
data/                 gitignored runtime data, see data/README.md
docs/                 methodology, week plan, proposal
```

## Datasets

**GNN**: the [Elliptic Bitcoin dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set),
a node-classification task run as binary licit vs illicit, with the unlabeled
nodes held out. The graph is sparse and split into 49 time steps with no edges
between them, so it is stored as a sparse adjacency and processed one time step
at a time. See `data/README.md` for sourcing.

**Diffusion**: 2D toy data generated locally, not downloaded. Six shape
generators are provided: spiral, two moons, checkerboard, heart, concentric
circles, and infinity. This is kept separate from the GNN data by design, since
classification and generation are different tasks and do not share inputs.

## Visualization

A raylib dashboard (under `app/gui/`) links the engine directly and runs it in
process. For the GNN it shows ranked accounts, per-account subgraphs, and a
top-k illicit subgraph, with gradient-based saliency standing in for a SHAP-style
explanation, computed by the engine's own `backward()`. For the diffusion model
it animates the reverse denoising process, a noise cloud resolving into the
chosen 2D shape, with a timestep slider and a faint overlay of the target shape.

## Team

Two teammates own bounded, independently testable tasks: the data pipeline, the
SGD and Adam optimizers, evaluation metrics, the toy-data generators, and loss
logging. See `docs/` for the full split. The engine core, the GNN forward and
backward logic, and the diffusion training loop are owned end to end by the
project lead.
