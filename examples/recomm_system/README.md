# Recommender System Experiments

**How to run:**  
```sh
cd examples/recomm_system/
python3 main.py
```

**Files:**
- `main.py` — Runs all experiments, calls each model script for various noise ratios, saves results to `experiment_results.csv`.
- `graph_nn.py` — Graph Neural Network (GCN) experiment.
- `graph_tm.py` — Graph Tsetlin Machine experiment.
- `tm_classifier.py` — Tsetlin Machine Classifier experiment.
- `prepare_dataset.py` — Dataset download, noise injection, preprocessing.
- `experiment_results.csv` — Results log (auto-generated).
- `test.ipynb` — Summarizes results, generates LaTeX tables.

