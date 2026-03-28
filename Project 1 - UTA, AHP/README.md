# Project 1 — UTA & AHP

Multi-Criteria Decision Analysis of European countries using the OECD Better Life Index.

## Project Structure

```
Project 1 - UTA, AHP/
├── data/
│   ├── raw/
│   │   └── kaggle_data.csv              # Raw OECD BLI dataset from Kaggle (long format)
│   ├── criteria_metadata.csv            # Criteria types: gain/cost for each criterion
│   ├── dataset.csv                      # Final MCDA dataset (26 countries x 8 criteria)
│   ├── dataset_description.md           # Answers to all 13 dataset description questions
│   └── european_capitals.json           # Coordinates of European capitals (for distance calc)
├── src/
│   ├── prepare_dataset.py               # Transforms raw data into final dataset.csv
│   └── find_dominated.py                # Finds all dominated alternatives in the dataset
├── Instructions/
│   ├── DA_Project_UTA_AHP.pdf           # Project requirements (UTA, AHP, grading)
│   ├── DA_dataset_description.pdf       # Dataset description requirements and questions
│   └── dataset_preparation_instructions.md  # Step-by-step data preparation guide
├── pyproject.toml
└── uv.lock
```
