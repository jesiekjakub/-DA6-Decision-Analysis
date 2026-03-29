# Project 1 — UTA & AHP

Multi-Criteria Decision Analysis of European countries using the OECD Better Life Index.

**Decision problem**: Selecting the best European country to live in from the perspective of a student based in Poznan, Poland.

**Dataset**: 26 European OECD members evaluated on 8 criteria (employment, earnings, health, satisfaction, work-life balance, air quality, distance).

## Quick Start

```bash
uv sync                # install dependencies
uv run jupyter lab     # open notebooks
```

**Start here**: Open [`reports/UTA_full_report.ipynb`](reports/UTA_full_report.ipynb) — the unified report covering the full dataset description, UTA 2.1 (inconsistency resolution), and UTA 2.2 (most discriminant value function).

## Project Structure

```
Project 1 - UTA, AHP/
├── reports/
│   ├── UTA_full_report.ipynb              # Unified report (dataset + UTA 2.1 + UTA 2.2)
│   └── dataset_description.md             # Answers to all 13 dataset description questions
├── data/
│   ├── raw/
│   │   └── kaggle_data.csv                # Raw OECD BLI dataset from Kaggle (long format)
│   ├── processed/
│   │   ├── dataset.csv                    # Final MCDA dataset (26 countries x 8 criteria)
│   │   ├── criteria_metadata.csv          # Criteria types: gain/cost for each criterion
│   │   └── european_capitals.json         # Capital coordinates (for distance calculation)
│   ├── preferences/
│   │   ├── preferences.csv                # 20 pairwise comparisons (DM preferential information)
│   │   └── selected_consistent_subset.csv # Indices of removed preferences for consistency
│   └── output/
│       └── marginal_value_functions.png   # Generated plot of marginal value functions
├── src/
│   ├── common/                            # Shared utilities used across all modules
│   │   ├── config.py                      # Project paths and UTA model constants
│   │   ├── data_loading.py                # Dataset, preferences, and removal indices loading
│   │   └── uta_core.py                    # Characteristic points, interpolation, constraints, solver
│   ├── uta_inconsistencies/               # UTA 2.1: Inconsistency resolution
│   │   ├── resolver.py                    # MILP for finding all minimal removal sets
│   │   └── UTA_inconsistencies_resolving.ipynb  # Step-by-step report for task 2.1
│   ├── uta_discrimination/                # UTA 2.2: Most discriminant value function
│   │   ├── solver.py                      # LP maximizing epsilon (discrimination threshold)
│   │   └── UTA_preference_model_solving.ipynb   # Step-by-step report for task 2.2
│   ├── prepare_dataset/
│   │   └── prepare_dataset.py             # Transforms raw OECD data into final dataset.csv
│   └── find_dominated/
│       └── find_dominated.py              # Identifies all Pareto-dominated alternatives
├── Instructions/
│   ├── DA_Project_UTA_AHP.pdf             # Project requirements (UTA, AHP, grading)
│   ├── DA_dataset_description.pdf         # Dataset description requirements and questions
│   └── da-lec3-notes.pdf                  # Lecture notes on UTA method
├── pyproject.toml
└── uv.lock
```

## Completed Tasks

- **Dataset**: 26 European countries, 8 criteria, all 13 description questions answered
- **UTA 2.1**: Inconsistency resolution — 20 pairwise comparisons with intentional Scandinavian cycle, 3 minimal removal sets found, Sweden > Norway removed
- **UTA 2.2**: Most discriminant value function — epsilon maximized, all equations/variables listed, marginal value function plots, full ranking of 26 countries
