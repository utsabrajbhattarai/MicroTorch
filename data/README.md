# data/

This folder is gitignored -- the Elliptic Bitcoin dataset isn't redistributed
in this repo. To set it up locally:

1. Download from Kaggle: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
   (requires a free Kaggle account)
2. Unzip into `data/raw/`, so you end up with:
   - `data/raw/elliptic_txs_features.csv`
   - `data/raw/elliptic_txs_edgelist.csv`
   - `data/raw/elliptic_txs_classes.csv`
3. Run the data-loading step (Teammate A's pipeline) to produce whatever
   gets written into `data/processed/` -- also gitignored.

Dataset facts worth keeping in mind while building the loader:
- 203,769 transaction nodes, 234,355 edges, 166 features per node.
- Only ~46,564 nodes are labeled (licit/illicit); the rest are unlabeled --
  train/eval only on the labeled subset, don't build anything that needs
  the unlabeled 77%.
- ~92% licit / ~8% illicit among labeled nodes (~1:11.5 imbalance).
- 49 separate timestep-graphs, no edges across timesteps -- this is what
  makes sparse per-timestep batching tractable on CPU (no dense
  203k x 203k adjacency matrix, ever).
- Published GCN baseline: ~95% AUROC. Use this as the external sanity
  check -- if your from-scratch GNN scores far above or below this on
  the first real run, suspect a bug before celebrating or panicking.

Diffusion's 2D toy data (spiral / two-moons) is *generated*, not
downloaded -- see Teammate B's toy-data generator, output also lands
(and stays gitignored) under `data/processed/`.
