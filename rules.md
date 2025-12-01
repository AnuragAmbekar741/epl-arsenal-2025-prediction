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
  - [ ] Install deep learning libraries: `tensorflow`, `keras` (for neural networks).
  - [ ] Add a `requirements.txt` or `pyproject.toml`.
- [ ] **Create initial notebooks / scripts**
  - [ ] `notebooks/01_exploration.ipynb`
  - [ ] `notebooks/02_modeling.ipynb`
  - [ ] `notebooks/03_deep_learning.ipynb` (NEW: for deep learning experiments)
  - [ ] `src/data_preprocessing.py`
  - [ ] `src/features.py`
  - [ ] `src/train_models.py`
  - [ ] `src/train_deep_learning.py` (NEW: for neural network training)
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

### 8.1 Traditional Machine Learning

- [ ] **Random Forest model**
  - [ ] Train a **Random Forest** classifier.
  - [ ] Perform basic hyperparameter tuning (e.g., n_estimators, max_depth).
  - [ ] Save the best model to `models/random_forest.pkl`.
- [ ] **(Optional but recommended) Gradient Boosting / XGBoost**
  - [ ] Train a gradient boosting model (e.g., XGBoost, LightGBM, or `sklearn.ensemble.GradientBoostingClassifier`).
  - [ ] Tune key hyperparameters.
  - [ ] Save model to `models/xgboost.pkl` (if implemented).

### 8.2 Deep Learning Models

- [ ] **Deep Learning: Feedforward Neural Network**

  - [ ] Set up TensorFlow/Keras environment.
  - [ ] Build a multi-layer feedforward neural network:
    - [ ] Input layer: number of features
    - [ ] Hidden layer 1: 64 neurons with ReLU activation
    - [ ] Hidden layer 2: 32 neurons with ReLU activation
    - [ ] Dropout layer (0.3) to prevent overfitting
    - [ ] Output layer: 3 neurons (Win/Draw/Loss) with softmax activation
  - [ ] Compile model with appropriate optimizer (e.g., Adam) and loss function (categorical_crossentropy).
  - [ ] Implement early stopping and model checkpointing.
  - [ ] Train model with validation split (e.g., 20%).
  - [ ] Save trained model to `models/neural_network.h5`.
  - [ ] Plot training history (loss and accuracy curves).
  - [ ] Evaluate on test set and compare with traditional ML models.

- [ ] **Deep Learning: LSTM (Long Short-Term Memory) Model**

  - [ ] Prepare sequential data:
    - [ ] Create sequences of last N matches (e.g., last 10 matches) for each prediction point.
    - [ ] Each sequence contains features from previous matches.
    - [ ] Ensure sequences only use past information (no data leakage).
  - [ ] Build LSTM model architecture:
    - [ ] Input layer: (sequence_length, n_features)
    - [ ] LSTM layer 1: 64 units, return_sequences=True
    - [ ] LSTM layer 2: 32 units
    - [ ] Dense layer: 16 neurons with ReLU activation
    - [ ] Output layer: 3 neurons (Win/Draw/Lraw/Loss) with softmax activation
  - [ ] Compile and train LSTM model.
  - [ ] Implement early stopping and validation monitoring.
  - [ ] Save trained model to `models/lstm_model.h5`.
  - [ ] Analyze what temporal patterns the LSTM learned.
  - [ ] Compare LSTM performance with feedforward network and traditional ML.

- [ ] **(Optional) Advanced: Hybrid LSTM + Feature Model**

  - [ ] Combine sequential patterns (LSTM) with match-specific features (Dense layers).
  - [ ] Architecture: Two parallel branches that merge before final prediction.
  - [ ] Evaluate if hybrid approach improves performance.

- [ ] **Deep Learning Model Comparison**
  - [ ] Compare all deep learning models:
    - [ ] Feedforward Neural Network
    - [ ] LSTM Model
    - [ ] (Optional) Hybrid Model
  - [ ] Create performance comparison table.
  - [ ] Analyze training curves and convergence behavior.
  - [ ] Document which architecture works best and why.

### 8.3 Comprehensive Model Comparison

- [ ] **Compare all models (Traditional + Deep Learning)**
  - [ ] Create a comprehensive comparison table:
    - [ ] Logistic Regression (baseline)
    - [ ] Random Forest
    - [ ] (Optional) XGBoost
    - [ ] Feedforward Neural Network
    - [ ] LSTM Model
  - [ ] Metrics to compare:
    - [ ] Accuracy
    - [ ] Precision, Recall, F1-score (per class)
    - [ ] Macro F1, Weighted F1
    - [ ] Training time
    - [ ] Model complexity
  - [ ] Visualize model performance comparison (bar charts, radar plots).
  - [ ] Analyze which model type performs best and discuss why.
  - [ ] Select the best-performing model as the **primary** model for predictions.

---

## 9. Model Evaluation & Interpretation

- [ ] **Evaluate on test set**
  - [ ] Use the reserved 2024–25 test set (when outcomes are available) OR use the latest available full season as test.
  - [ ] Generate for all models:
    - [ ] Confusion matrix
    - [ ] Classification report
    - [ ] Calibration of predicted probabilities (optional).
- [ ] **Feature importance / interpretability**
  - [ ] For tree-based models:
    - [ ] Compute and plot feature importances.
  - [ ] For logistic regression:
    - [ ] Inspect coefficients and their signs.
  - [ ] For deep learning models:
    - [ ] Use techniques like SHAP values or permutation importance (if applicable).
    - [ ] Analyze which features the neural network focuses on.
    - [ ] Visualize learned patterns (e.g., attention weights for LSTM if using attention).
  - [ ] Interpret what features typically increase the chance of Win vs Loss across all model types.
- [ ] **Deep Learning Specific Analysis**
  - [ ] Plot training/validation loss and accuracy curves for neural networks.
  - [ ] Analyze overfitting/underfitting (compare training vs validation performance).
  - [ ] For LSTM: Analyze what sequence patterns it learned (e.g., does it capture momentum?).
  - [ ] Compare model complexity vs performance trade-offs.

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
    - [ ] Model selection and training procedure:
      - [ ] Traditional ML models (Logistic Regression, Random Forest)
      - [ ] Deep learning models (Feedforward NN, LSTM)
      - [ ] Architecture choices and hyperparameters
    - [ ] Evaluation metrics and results:
      - [ ] Comparison of all models
      - [ ] Discussion of why deep learning models perform better/worse
    - [ ] Interpretation of key features:
      - [ ] Traditional ML feature importance
      - [ ] Deep learning learned patterns
    - [ ] Research findings:
      - [ ] Answers to research questions
      - [ ] Novel insights from deep learning approach
    - [ ] Limitations and future work:
      - [ ] Potential improvements (e.g., transformer models, attention mechanisms)
      - [ ] Adding player/tactical data
      - [ ] Real-time prediction system
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

## 8.5 Deep Learning Research Questions

- [ ] **Formulate research questions**
  - [ ] Do deep learning models outperform traditional ML for EPL match prediction?
  - [ ] Does LSTM capture temporal patterns better than feedforward networks?
  - [ ] What sequence length (last N matches) works best for LSTM?
  - [ ] How do deep learning models handle class imbalance (Win/Draw/Loss)?
  - [ ] What features are most important for each model type?
- [ ] **Document findings**
  - [ ] Answer each research question with evidence from experiments.
  - [ ] Include in final report with supporting visualizations and metrics.

---

## Completion Criteria

The project is considered **complete** when:

- [ ] A cleaned and feature-engineered dataset exists in `data/processed/`.
- [ ] At least two traditional ML models (e.g., Logistic Regression and Random Forest) are trained and evaluated.
- [ ] At least two deep learning models (Feedforward Neural Network and LSTM) are trained and evaluated.
- [ ] All models are compared comprehensively with performance metrics and visualizations.
- [ ] Research questions about deep learning effectiveness are answered with evidence.
- [ ] Predictions for Arsenal's fixtures are generated using the best-performing model.
- [ ] Visualizations of performance and predictions are created.
- [ ] A final written report and presentation slide deck are completed and stored in `reports/`.
- [ ] Repository is organized, documented, and reproducible.
- [ ] Deep learning models are saved and can be loaded for inference.

---

## Summary of changes

1. Section 1: Added TensorFlow/Keras and new files
2. Section 8: Expanded with deep learning subsections
3. New Section 8.5: Research questions
4. Section 9: Added deep learning evaluation methods
5. Section 12: Added deep learning discussion to the report
6. Completion Criteria: Updated to require deep learning models

---

## What this adds

- Feedforward Neural Network (required)
- LSTM Model (required)
- Hybrid Model (optional, advanced)
- Research questions to answer
- Comprehensive model comparison
- Deep learning-specific evaluation

---
