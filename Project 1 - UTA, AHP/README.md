# Project 1 — UTA & AHP

Multi-Criteria Decision Analysis of European countries using the OECD Better Life Index.

## Project Structure

```
Project 1 - UTA, AHP/
├── data/
│   ├── raw/
│   │   └── kaggle_data.csv                  # Raw OECD BLI dataset from Kaggle (long format)
│   ├── criteria_metadata.csv                # Criteria types: gain/cost for each criterion
│   ├── dataset.csv                          # Final MCDA dataset (26 countries x 8 criteria)
│   ├── dataset_description.md               # Answers to all 13 dataset description questions
│   ├── european_capitals.json               # Coordinates of European capitals (for distance calc)
│   ├── preferences.csv                      # 20 pairwise comparisons (DM preferential information)
│   ├── selected_consistent_subset.csv       # Indices of preferences removed for consistency
│   └── marginal_value_functions.png         # Plot of marginal value functions (generated)
├── src/
│   ├── prepare_dataset.py                   # Transforms raw data into final dataset.csv
│   ├── find_dominated.py                    # Finds all dominated alternatives in the dataset
│   ├── UTA_inconsistencies_resolving/
│   │   ├── uta.py                           # UTA 2.1: inconsistency resolution (MILP solver)
│   │   └── UTA_inconsistencies_resolving.ipynb  # Report: process and results discussion
│   └── UTA_preference_model_solving/
│       ├── uta_solver.py                    # UTA 2.2: most discriminant value function (max ε)
│       └── UTA_preference_model_solving.ipynb   # Report: plots, ranking, and discussion
├── Instructions/
│   ├── DA_Project_UTA_AHP.pdf               # Project requirements (UTA, AHP, grading)
│   └── DA_dataset_description.pdf           # Dataset description requirements and questions
├── pyproject.toml
└── uv.lock
```
