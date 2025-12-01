# Project Rules & Task Checklist

_Predicting Arsenal’s EPL Match Outcomes Using Supervised ML_

This document defines the rules and ordered tasks required to complete the project from start to finish.

---

## 0. General Rules

- Use **Python** as the main language.
- Use **Git + GitHub** for version control. Commit at each major step.
- Keep all data-related code in a `notebooks/` or `src/` folder and configuration in a separate `config/` or `.env` file.
- Use **reproducible** workflows:
  - Fix random seeds where possible.
  - Save preprocessed datasets and trained models to disk.
- Document decisions in a `LOG.md` as you go (data choices, feature choices, model changes, etc.).

---

## 1. Project Setup

- [ ] **Create repository structure**
  - [ ] Initialize Git repository.
  - [ ] Create folders: `data/raw/`, `data/processed/`, `src/`, `notebooks/`, `models/`, `reports/`, `figures/`.
  - [ ] Add a `README.md` with a short project description and goals.
- [ ] **Set up Python environment**
  - [ ] Create a virtual environment (e.g., `venv` or `conda`).
  - [ ] Install required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` (optional), `jupyter`.
  - [ ] Add a `requirements.txt` or `pyproject.toml`.
- [ ] **Create initial notebooks / scripts**
  - [ ] `notebooks/01_exploration.ipynb`
  - [ ] `notebooks/02_modeling.ipynb`
  - [ ] `src/data_preprocessing.py`
  - [ ] `src/features.py`
  - [ ] `src/train_models.py`
  - [ ] `src/evaluate.py`

---

## 2. Data Collection

- [ ] **Identify data sources**
  - [ ] Confirm EPL historical results source(s) (e.g., Football-Data.co.uk, Kaggle, FBRef).
  - [ ] Ensure coverage for ~10–15 seasons (2010–2024) including the 2024–25 fixtures list for Arsenal.
- [ ] **Download datasets**
  - [ ] Download all relevant CSV files for EPL seasons (match results, team stats).
  - [ ] Save raw files in `data/raw/`.
- [ ] **Document data sources**
  - [ ] In `data/README.md`, list:
    - Data source URLs
    - Seasons covered
    - Important columns (e.g., HomeTeam, AwayTeam, FTHG, FTAG, FTR, shots, cards, xG if available).

---

## 3. Data Understanding & Cleaning

- [ ] **Initial data inspection**
  - [ ] Load all seasons into a notebook (`01_exploration.ipynb`).
  - [ ] Inspect columns, datatypes, missing values, and basic summary statistics.
- [ ] **Filter for EPL and Arsenal-related matches**
  - [ ] Keep only English Premier League matches (if other leagues present).
  - [ ] Mark which rows correspond to Arsenal matches (either as home or away).
- [ ] **Standardize team names**
  - [ ] Ensure consistent naming of Arsenal and all opponents across seasons.
- [ ] **Handle missing values**
  - [ ] Decide how to handle missing stats (e.g., drop, impute, or ignore certain columns).
  - [ ] Implement cleaning logic in `src/data_preprocessing.py`.
- [ ] **Create unified master table**
  - [ ] Concatenate all seasons into one DataFrame.
  - [ ] Save cleaned master data to `data/processed/epl_cleaned.csv`.

---

## 4. Target Definition & Labeling

- [ ] **Define match outcome labels**
  - [ ] For each Arsenal match, create a target label:
    - `Win`, `Draw`, `Loss` (multiclass).
  - [ ] Use final scores (home/away) and which team is Arsenal to derive labels.
- [ ] **Add basic match metadata**
  - [ ] Include: season, date, home/away flag, opponent team, final score.
  - [ ] Save labeled data to `data/processed/arsenal_labeled.csv`.

---

## 5. Feature Engineering

- [ ] **Design feature set**
  - [ ] Match-level features:
    - [ ] Goals scored and conceded in previous N matches.
    - [ ] Rolling averages (e.g., last 5 matches) of:
      - Goals scored
      - Goals conceded
      - Points (win=3, draw=1, loss=0)
    - [ ] Home vs away indicator.
    - [ ] Opponent’s league position before the match.
    - [ ] Goal difference, shot difference, xG-related stats if available.
  - [ ] Season/temporal features:
    - [ ] Matchday number
    - [ ] Days since last match
- [ ] **Implement feature creation**
  - [ ] In `src/features.py`, implement rolling window computations.
  - [ ] Make sure features only use information from **past** matches (no data leakage).
- [ ] **Generate final feature matrix**
  - [ ] Combine all engineered features with the target label.
  - [ ] Save to `data/processed/arsenal_features.csv`.

---

## 6. Train/Test Split Strategy

- [ ] **Define training and test periods**
  - [ ] Use older seasons (e.g., 2010–2023/24) as training data.
  - [ ] Reserve the 2024–25 season (or its part) as test data.
- [ ] **Implement split**
  - [ ] In `src/train_models.py`, create a time-based split:
    - `X_train`, `y_train` (past seasons)
    - `X_test`, `y_test` (2024–25 fixtures with known outcomes once available)
- [ ] **Optionally create validation set**
  - [ ] Use a rolling-origin or season-wise split for validation (e.g., train on seasons up to 2018, validate on 2019, etc.).

---

## 7. Baseline Modeling

- [ ] **Baseline model setup**
  - [ ] Start with a simple **Logistic Regression** classifier.
  - [ ] Standardize/normalize features as needed.
- [ ] **Train baseline**
  - [ ] Fit Logistic Regression on training data.
  - [ ] Save the trained model in `models/logreg_baseline.pkl`.
- [ ] **Evaluate baseline**
  - [ ] Compute Accuracy, Precision, Recall, F1-score per class, and Confusion Matrix.
  - [ ] Log results in `reports/metrics_baseline.md`.

---

## 8. Advanced Models

- [ ] **Random Forest model**
  - [ ] Train a **Random Forest** classifier.
  - [ ] Perform basic hyperparameter tuning (e.g., n_estimators, max_depth).
  - [ ] Save the best model to `models/random_forest.pkl`.
- [ ] **(Optional but recommended) Gradient Boosting / XGBoost**
  - [ ] Train a gradient boosting model (e.g., XGBoost, LightGBM, or `sklearn.ensemble.GradientBoostingClassifier`).
  - [ ] Tune key hyperparameters.
- [ ] **(Optional) Deep Learning model**
  - [ ] Build a simple feedforward neural network (e.g., with Keras/PyTorch).
  - [ ] Use the same feature matrix as input.
  - [ ] Evaluate whether it outperforms traditional models.
- [ ] **Compare models**
  - [ ] Create a table of metrics for all models: Accuracy, F1, macro F1, weighted F1.
  - [ ] Select the best-performing and most interpretable model as the **primary** model.

---

## 9. Model Evaluation & Interpretation

- [ ] **Evaluate on test set**
  - [ ] Use the reserved 2024–25 test set (when outcomes are available) OR use the latest available full season as test.
  - [ ] Generate:
    - Confusion matrix
    - Classification report
    - Calibration of predicted probabilities (optional).
- [ ] **Feature importance / interpretability**
  - [ ] For tree-based models:
    - [ ] Compute and plot feature importances.
  - [ ] For logistic regression:
    - [ ] Inspect coefficients and their signs.
  - [ ] Interpret what features typically increase the chance of Win vs Loss.

---

## 10. Predictions for Upcoming Fixtures

- [ ] **Prepare future fixtures**
  - [ ] Obtain 2024–25 Arsenal fixtures (opponent, date, home/away).
  - [ ] Construct feature values for upcoming matches based on most recent historical data.
- [ ] **Generate predictions**
  - [ ] Use the final model to:
    - [ ] Output probabilities: P(Win), P(Draw), P(Loss) for each fixture.
  - [ ] Save predictions to `reports/arsenal_2024_25_predictions.csv`.

---

## 11. Visualization

- [ ] **Performance visualizations**
  - [ ] Plot confusion matrix as heatmap.
  - [ ] Plot model accuracy / F1 across seasons (if using season-wise evaluation).
- [ ] **Prediction visualizations**
  - [ ] Plot Arsenal fixture list with predicted probabilities as bar charts or line plots.
  - [ ] Optionally highlight high-risk (high predicted loss) matches.
- [ ] **Save figures**
  - [ ] Save all plots into `figures/` with descriptive filenames.

---

## 12. Documentation & Reporting

- [ ] **Technical report**
  - [ ] Write a structured report (`reports/final_report.md` or `.pdf`) including:
    - [ ] Introduction and motivation
    - [ ] Data sources and preprocessing steps
    - [ ] Feature engineering methodology
    - [ ] Model selection and training procedure
    - [ ] Evaluation metrics and results
    - [ ] Interpretation of key features
    - [ ] Limitations and future work (e.g., adding player/tactical data, advanced deep learning).
- [ ] **Code documentation**
  - [ ] Add docstrings and comments in `src/` files.
  - [ ] Update `README.md` with:
    - [ ] How to run preprocessing
    - [ ] How to train models
    - [ ] How to reproduce evaluation and prediction steps.

---

## 13. Final Presentation

- [ ] **Prepare slides**
  - [ ] Summarize:
    - Problem statement
    - Data and features
    - Model comparisons
    - Key findings
    - Example predictions for specific matches
  - [ ] Include key plots (confusion matrix, probability charts).
- [ ] **Rehearse and finalize**
  - [ ] Ensure slides tell a coherent story from motivation → method → results → implications.

---

## Completion Criteria

The project is considered **complete** when:

- [ ] A cleaned and feature-engineered dataset exists in `data/processed/`.
- [ ] At least two ML models (e.g., Logistic Regression and Random Forest) are trained and evaluated.
- [ ] Predictions for Arsenal’s fixtures are generated and saved.
- [ ] Visualizations of performance and predictions are created.
- [ ] A final written report and presentation slide deck are completed and stored in `reports/`.
- [ ] Repository is organized, documented, and reproducible.
