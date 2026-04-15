# Project 2 — Preference Learning

Decision Analysis course, Project 2. A comparative study of three preference
learning methods on the `lectures evaluation` dataset: a monotone-constrained
gradient boosted tree, a fully interpretable neural MCDA model, and a
conventional deep MLP. Everything lives in a single report notebook.

## Report

| File | Models | Role |
|---|---|---|
| [notebooks/report.ipynb](notebooks/report.ipynb) | XGBoost (monotone), ANN-UTADIS, Deep MLP | Full project report — dataset, three models, sections 2.1 and 2.2 for each, final comparison |
| [reports/report.html](reports/report.html) | — | HTML export of the notebook |

All three models use the same dataset, the same class merging, and the same
train/test split (`random_state = 1234`, 80/20 stratified). They are meant to
be read in order.

## Dataset

`data/lectures evaluation.csv` — 1000 alternatives, 4 normalized criteria
$(c_1, \dots, c_4)$ with ordinal values in $\{0, 0.25, 0.5, 0.75, 1.0\}$ and
a quality class in $\{0, 1, 2, 3, 4\}$. The original 5 classes are imbalanced
(class 4 has only 27 samples) so they are merged into three:
$\{0, 1\} \to$ low, $\{2\} \to$ medium, $\{3, 4\} \to$ high.

## Project layout

```
Project 2 - Preference Learning/
├── data/
│   └── lectures evaluation.csv    # input dataset (no header)
├── Instructions/
│   ├── DA_preference_learning_project.pdf
│   └── da-lec6-notes.pdf
├── layers/                        # ANN-UTADIS PyTorch building blocks
│   ├── __init__.py
│   ├── uta.py                     # additive value-function wrapper
│   ├── monotonic_layer.py         # spread → activation → combine block
│   ├── criterion_layer_spread.py
│   ├── criterion_layer_combine.py
│   ├── leaky_hard_sigmoid.py
│   ├── norm_layer.py              # U(0) = 0, U(1) = 1 anchoring
│   └── threshold_layer.py         # binary + ordinal K-1 threshold layers
├── notebooks/
│   └── report.ipynb               # single consolidated report
├── reports/
│   └── report.html                # HTML export
├── pyproject.toml
├── uv.lock
└── .python-version
```

## Running

```bash
uv sync                         # install all dependencies
uv run jupyter lab              # open notebooks/report.ipynb
```

To export the report to a self-contained HTML file:

```bash
uv run jupyter nbconvert --to html --output-dir reports notebooks/report.ipynb
```

The notebook is runnable top-to-bottom from a clean kernel. All random seeds
are pinned so metrics are reproducible.

## What the report covers

The report is organized as four sections:

1. **Dataset** — loading, class merging, train/test split.
2. **XGBoost with monotone constraints** — the interpretable ML baseline.
3. **ANN-UTADIS** — a neural network with 50 hidden components per criterion,
   cumulative-sigmoid ordinal thresholds, and a `NormLayer` that anchors
   $U(0) = 0$ and $U(1) = 1$ so the learned marginal utility functions are
   directly readable.
4. **Deep MLP** — a plain $4 \to 64 \to 32 \to 16 \to 3$ ReLU network with
   dropout and early stopping.

Every model has the same sub-structure:

- training + metrics (Accuracy, weighted F1, one-vs-rest AUC, rounded to four
  decimal places);
- confusion matrix + description;
- **Section 2.1 — Decision explanation** for alternatives 574, 757 and 962:
  per-criterion contributions (or SHAP waterfalls), minimum single-criterion
  change — both analytical and empirical;
- **Section 2.2 — Model interpretation**: criterion influence, nature
  (gain / cost / non-monotonic), preference thresholds, indifference
  regions, dependencies, and a post-hoc feature importance technique (PDP
  and/or PFI);
- short summary.

The report ends with a comparison section that collects all three metric
sets into one table, plots them side by side, and concludes on the accuracy
vs interpretability trade-off.
