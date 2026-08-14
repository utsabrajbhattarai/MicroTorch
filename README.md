# MicroTorch

A from-scratch C++ automatic-differentiation engine with built-in gradient
verification, demonstrated by a graph neural network for anti-money-laundering
detection and a diffusion-based generative model. No machine-learning libraries
are used — only Eigen for dense and sparse linear algebra.

The engine is the product. The GNN and the diffusion model are demos that prove
it is a genuine framework rather than a one-architecture hack: both are built
from the same minimal op set, with no engine-side special-casing for either.

## Status

Complete and verified end to end. The `Tensor` type and reverse-mode autograd
work through a topological backward pass, every operation is checked against
finite differences, and both demos train and produce results.

**Engine**
- `Tensor` and reverse-mode autograd (topological backward pass over a graph of
  captured backward closures)
- Operations, each gradient-checked: `matmul`, `broadcast_add`, `add`, `relu`,
  `sum`, `mean`, `mse_loss`, `neighbor_aggregation` (sparse), and the fused
  `softmax_cross_entropy` (with a masked overload for semi-supervised labels)
- Optimizers: `SGD` and `Adam`, validated on a toy quadratic
- A `gradient_check` harness with a Catch2 test per op — including a
  deliberately broken op, so the suite proves the harness *catches* wrong
  gradients, not just that it passes correct ones

**Demos**
- **GNN** (`gnn_train`): loads the Elliptic dataset, builds a two-layer GCN from
  the ops above, trains with Adam on a stratified split, and reports
  precision / recall / F1 / accuracy / AUROC. Runs a GNN-vs-MLP ablation across
  three seeds to isolate the contribution of graph structure; both land around
  0.96–0.98 AUROC, in line with the published ~0.95 GCN baseline.
- **Diffusion** (`diffusion_train`): a DDPM denoiser (an MLP with a sinusoidal
  time embedding) trained with MSE on predicted noise, then ancestral sampling
  that resolves Gaussian noise into a 2D shape and exports the frames.

**Tooling**
- Evaluation metrics, a CSV training logger, and two raylib front-ends: an
  anti-money-laundering dashboard and a diffusion animation viewer.

## Build

Requires CMake 3.16+ and a C++17 compiler. Eigen and Catch2 are pulled through
CMake `FetchContent`, so no manual install is needed. raylib is fetched only for
the GUI executables — the engine and its tests do not depend on it.

```bash
cmake -B build -S .
cmake --build build
ctest --test-dir build
```

`ctest` runs the full suite. Every operation carries a gradient check, so a
wrong backward pass shows up as a named test failure rather than a silent bug
that only surfaces later as a model that never learns.

## Run the demos

```bash
# GNN: trains + evaluates, prints metrics and the GNN-vs-MLP ablation
./build/app/gnn_train

# Diffusion: trains the denoiser, samples, writes frames.csv + generated.csv
./build/app/diffusion_train

# GUIs — run from the repo root so they find their CSV artifacts
./build/app/gui/gnn_gui            # AML dashboard   <- gui_artifacts/*.csv
./build/app/gui/diffusion_viewer   # animation       <- frames/*.csv
```

## Layout

```
include/microtorch/   public headers
  Tensor.hpp          the core tensor type + autograd node
  gradient_check.hpp  finite-difference verification
  ops/                matmul, broadcast_add, add, relu, reduction,
                      mse_loss, neighbor_aggregation, softmax_cross_entropy
  optim/              sgd, adam
  gnn/                node-classification model
  diffusion/          denoiser + noise schedule
  data/               data_loader (Elliptic) + toy_data (2D shapes)
  eval/               precision / recall / F1 / accuracy / AUROC
  gui/                CSV export for the dashboard
src/                  engine implementation, builds as libmicrotorch
tests/                Catch2, mirrors src one-to-one, a test file per op
app/                  gnn_train, diffusion_train
  gui/                gnn_dashboard + diffusion_viewer (raylib) + csv_loader
data/                 gitignored dataset, see data/README.md
gui_artifacts/        CSVs the dashboard reads (nodes, edges, accounts, metrics)
frames/               pre-baked diffusion animations for the viewer
```

## Datasets

**GNN**: the [Elliptic Bitcoin dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set),
a node-classification task run as binary licit vs illicit with the unlabeled
nodes held out. The graph is sparse and split into 49 time steps with no edges
between them, so it is stored as a sparse adjacency — self-looped and
symmetrically normalized (`D^-1/2 A D^-1/2`) — and trained on the labeled
subset. See `data/README.md` for sourcing.

**Diffusion**: 2D toy data generated locally, not downloaded. Seven shape
generators are provided: spiral, light spiral, two moons, checkerboard, heart,
concentric circles, and infinity. This is kept separate from the GNN data by
design, since classification and generation are different tasks and share no
inputs.

## Visualization

Two raylib front-ends under `app/gui/`, deliberately decoupled from the engine:
they read CSV artifacts on disk rather than linking `libmicrotorch`, so a rebuild
of the engine never breaks the GUI and vice versa.

- **AML dashboard** (`gnn_gui`): reads `gui_artifacts/*.csv` and shows a data
  preview, a paginated table of accounts ranked by predicted risk, a metrics
  panel, and an interactive per-account subgraph — counterparties split into
  received (left) and sent (right) with directional arrows, colored by predicted
  label.
- **Diffusion viewer** (`diffusion_viewer`): reads `frames/*.csv`, offers a menu
  of shapes, and animates the reverse denoising process — a noise cloud
  resolving into the chosen 2D shape — with a scrub slider, play/pause, and a
  faint overlay of the target shape.

## Team

Two teammates own bounded, independently testable pieces: the data pipeline, the
SGD and Adam optimizers, evaluation metrics, the toy-data generators, and loss
logging. The engine core, the GNN forward and backward logic, and the diffusion
training loop are owned end to end by the project lead.
