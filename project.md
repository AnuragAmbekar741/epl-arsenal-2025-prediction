## How to Navigate the Project and Run the Code

### Project Structure

The repository is organized as follows:

- **`data/`**

  - **`data/raw/`**: Original CSVs for each EPL season (e.g., `epl-2000-01.csv`, `epl-arsenal-2025-26.csv`).
  - **`data/processed/`**: Cleaned and engineered datasets produced by the pipeline, e.g.:
    - `epl_cleaned.csv` – cleaned league-wide match data
    - `arsenal_labeled.csv` – Arsenal-only matches with Win/Draw/Loss labels
    - `league_standings.csv` – final table for each team and season
    - `arsenal_features.csv` – final feature matrix used for modeling

- **`src/`** – Core Python modules (script versions of the pipeline):

  - **`data_preprocessing.py`** – loads raw EPL CSVs, cleans data, standardizes team names, converts dates, filters Arsenal matches, and computes league standings for all seasons.
  - **`features.py`** – builds all engineered features for Arsenal matches (rolling stats, opponent features, shot conversion, etc.) and saves `arsenal_features.csv`.
  - **`train_models.py`** – trains all models (Logistic Regression, Random Forest, Neural Networks, LSTM, Hybrid LSTM), performs time-based train/test split, and saves trained models plus `reports/model_comparison.csv`.
  - **`evaluate.py`** – loads trained models, evaluates them on the 2024–25 season, and saves confusion matrices, ROC curves, and precision–recall curves in `figures/`.
  - **`feature_importance.py`** – analyzes feature importance for Random Forest, Logistic Regression, and the Neural Network, and generates comparison plots + a text interpretation.
  - **`predict_fixtures.py`** – uses the best model (Neural Network with class weights) to predict upcoming Arsenal fixtures and produces prediction CSVs and plots.

- **`notebooks/`** – Jupyter notebooks (interactive versions of the main steps):

  - `01_data_preprocessing.ipynb` – run and inspect the preprocessing step.
  - `02_features.ipynb` – feature engineering.
  - `03_train_models.ipynb` – model training.
  - `04_evaluate.ipynb` – model evaluation.
  - `05_feature_importance.ipynb` – feature importance analysis.
  - `06_predict_fixtures.ipynb` – predictions for future fixtures.
  - (Plus earlier exploration notebooks, e.g. `01_exploration.ipynb`, `02_feature_exploration.ipynb`.)

- **`models/`**  
  Saved models and preprocessing objects:

  - `logreg_baseline.pkl`, `random_forest.pkl`
  - `neural_network.h5`, `neural_network_preprocessing.pkl`
  - `neural_network_weighted.h5`, `neural_network_weighted_preprocessing.pkl`
  - `lstm_model.h5`, `lstm_preprocessing.pkl`
  - `hybrid_lstm.h5`, `hybrid_lstm_preprocessing.pkl`
  - Training history plots (e.g. `neural_network_weighted_training_history.png`)

- **`figures/`**  
  Plots for:

  - Training history (loss/accuracy)
  - Confusion matrices
  - ROC and Precision–Recall curves
  - Feature importance comparisons
  - Future fixture prediction visuals

- **`reports/`**  
  Tabular outputs and text reports:

  - `model_comparison.csv` – accuracy and per-class metrics for all models
  - `model_evaluation_comparison.csv` – ROC AUC comparison across models
  - `arsenal_2024_25_predictions.csv` – predicted outcomes and probabilities for future fixtures
  - `feature_importance_interpretation.txt` – written summary of key features

- **`project.md`** – This document (how to navigate and run the project).
- **`README.md`** – High-level project overview (short version).

---

## Environment Setup

1. **Create a Python environment** (example using `conda`, adjust as needed):

```bash
cd epl-arsenal-prediction

conda create -n arsenal-ml python=3.10
conda activate arsenal-ml
```

2. **Install dependencies** (install these packages manually if you don’t have a `requirements.txt`):

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn joblib
```

(If you add back a `requirements.txt`, you can instead run `pip install -r requirements.txt`.)

---

## Running the Pipeline via Python Scripts

You can run the full pipeline in 5 main steps from the command line.

### 1. Data Preprocessing

**Goal**: Clean all raw EPL data and generate Arsenal-only labeled matches plus league standings.

```bash
cd /Users/anuragambekar/Desktop/epl-arsenal-prediction

python src/data_preprocessing.py
```

Outputs (in `data/processed/`):

- `epl_cleaned.csv`
- `arsenal_labeled.csv`
- `league_standings.csv`

### 2. Feature Engineering

**Goal**: Build leakage-safe features for each Arsenal match.

```bash
python src/features.py
```

Output:

- `data/processed/arsenal_features.csv`

### 3. Model Training

**Goal**: Train all models using a time-based split (train on earlier seasons, test on 2024–25).

```bash
python src/train_models.py
```

Outputs:

- Trained models in `models/` (`.pkl` and `.h5` files)
- Training history plots in `figures/`
- `reports/model_comparison.csv` summarizing performance of all models
  - Best model: Neural Network with class weights (`neural_network_weighted.h5`)

### 4. Model Evaluation

**Goal**: Generate detailed evaluation plots and metrics for selected models on the 2024–25 test season.

```bash
python src/evaluate.py
```

Outputs (in `figures/`):

- Confusion matrices per model
- ROC curves and AUC scores
- Precision–Recall curves

Also:

- `figures/model_evaluation_comparison.csv` – AUC comparison across models

### 5. Future Fixtures Prediction

**Goal**: Use the best model to predict future Arsenal fixtures.

Prerequisites:

- `models/neural_network_weighted.h5` and `models/neural_network_weighted_preprocessing.pkl` (from training step)
- Historical features and matches:
  - `data/processed/arsenal_features.csv`
  - `data/processed/arsenal_labeled.csv`
- A fixtures file, e.g. `epl-arsenal-2025-26.csv`, placed at the project root (path used in the script).

Run:

```bash
python src/predict_fixtures.py
```

Outputs:

- `reports/arsenal_2024_25_predictions.csv` – date, opponent, venue, predicted result, and W/D/L probabilities
- `figures/arsenal_fixtures_predictions.png` – stacked bar chart of probabilities per fixture
- `figures/arsenal_predictions_summary.png` – distribution of predicted results and confidence

---

## Using the Jupyter Notebooks Instead

If you prefer an interactive, step-by-step run:

1. Start Jupyter:

```bash
cd /Users/anuragambekar/Desktop/epl-arsenal-prediction
jupyter lab
# or
jupyter notebook
```

2. Open and run notebooks in order:

- **`01_data_preprocessing.ipynb`** – run all cells to reproduce preprocessing.
- **`02_features.ipynb`** – create and save feature matrix.
- **`03_train_models.ipynb`** – train all models and generate `model_comparison.csv`.
- **`04_evaluate.ipynb`** – evaluate models and generate plots.
- **`05_feature_importance.ipynb`** – analyze feature importance and save plots + text report.
- **`06_predict_fixtures.ipynb`** – construct features for future fixtures and generate predictions.

You can also use:

- `01_exploration.ipynb` and `02_feature_exploration.ipynb` (if present) for exploratory data analysis and understanding distributions, correlations, etc.

---

## What to Look At for Specific Questions

- **Data Cleaning / Standings logic**

  - `src/data_preprocessing.py`
  - `01_data_preprocessing.ipynb`

- **Feature Definitions (what each feature means)**

  - `src/features.py`
  - `02_features.ipynb`

- **Model Architectures and Hyperparameters**

  - `src/train_models.py` (Logistic Regression, Random Forest, NNs, LSTM, Hybrid LSTM)
  - `03_train_models.ipynb`

- **Evaluation Metrics and Plots**

  - `src/evaluate.py`
  - `reports/model_comparison.csv`
  - `figures/*_confusion_matrix.png`, `*_roc_curves.png`, `*_precision_recall.png`
  - `04_evaluate.ipynb`

- **Feature Importance and Interpretation**

  - `src/feature_importance.py`
  - `05_feature_importance.ipynb`
  - `figures/feature_importance_comparison.png`
  - `reports/feature_importance_interpretation.txt`

- **Future Fixture Predictions**
  - `src/predict_fixtures.py`
  - `reports/arsenal_2024_25_predictions.csv`
  - `figures/arsenal_fixtures_predictions.png`, `arsenal_predictions_summary.png`
  - `06_predict_fixtures.ipynb`

---

## Tips / Common Issues

- **Missing dependencies**: If a script fails with `ModuleNotFoundError` (e.g., for `tensorflow`), install the missing package with `pip install package-name` and rerun.
- **File not found (CSV)**: Ensure all raw season files are in `data/raw/` with names like `epl-YYYY-YY.csv` and that any fixtures file paths in `predict_fixtures.py` or the notebook are correct.
- **TensorFlow not installed**: The code will skip deep learning models if TensorFlow is unavailable. For full functionality, install TensorFlow (CPU version is usually enough).

You can adjust paths (`../data/...`) if you run scripts from a different working directory, but running everything from the project root is recommended.
