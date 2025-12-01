# Project Log

This file documents key decisions, changes, and insights throughout the project.

## Date: January 2025

### Project Setup

- Repository structure created
- Conda environment configured (epl-arsenal)
- Initial code files created with TODO placeholders
- Using hybrid approach: Jupyter notebooks for exploration, Python scripts for production code

### Decisions Made

- Using conda for environment management (easier for data science packages)
- Using scikit-learn for baseline and Random Forest models
- **ADDED: Deep learning models** (Feedforward Neural Network and LSTM) for research-oriented approach
- Time-based train/test split strategy (important to avoid data leakage)
- Feature engineering will use rolling windows of 5 matches

### Environment Setup

- Python 3.10
- Core libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter
- **Deep learning libraries added**: TensorFlow 2.20.0, Keras 3.12.0
- All libraries installed and verified

---

## Data Collection & Exploration Phase

### Data Sources

- **Source**: Historical EPL match data (2000-2025 seasons)
- **Files**: 26 CSV files (one per season)
- **Naming convention**: `epl-YYYY-YY.csv` (e.g., `epl-2024-25.csv`)
- **Location**: `data/raw/`

### Data Exploration Findings (from `01_exploration.ipynb`)

#### Data Loading Status

- **Total files**: 26 CSV files
- **Successfully loaded**: 25 files (96% success rate)
- **Problematic files**:
  - `epl-2003-04.csv`: Parsing error (line 305 has extra commas) - **Partially fixed**: 335/380 matches loaded
  - `epl-2004-05.csv`: Encoding + parsing errors - **Needs fix in preprocessing phase**

#### Data Structure

- **Total matches loaded**: ~8,861 matches across 25 seasons
- **Arsenal matches found**: 919 matches
- **Columns**: 213 columns (varies by season due to different data sources)
- **Seasons covered**: 2000-01 to 2025-26 (missing 2003-04 and 2004-05 fully)

#### Data Quality Issues Identified

1. **Missing Values**:

   - 212 columns have missing values
   - Many betting columns have high missing rates (37% for some columns)
   - Some columns are completely empty (Unnamed columns)
   - **Action**: Will handle in preprocessing phase

2. **Date Format**:

   - Dates are strings in format "DD/MM/YY" (e.g., "19/08/00")
   - Need to convert to datetime format
   - **Action**: Will standardize in preprocessing phase

3. **Team Name Consistency**:

   - 46 unique team names found across all seasons
   - Need to check for variations (e.g., "Man United" vs "Manchester United")
   - Arsenal name is consistent: "Arsenal"
   - **Action**: Will standardize team names in preprocessing phase

4. **Column Variations**:

   - Different seasons have different columns
   - Some columns appear in some seasons but not others
   - Empty/unnamed columns need to be removed
   - **Action**: Will align columns and remove empty ones in preprocessing

5. **File-Specific Issues**:
   - `epl-2003-04.csv`: Malformed lines with extra commas
   - `epl-2004-05.csv`: Encoding issues (not UTF-8) + parsing errors
   - **Action**: Will implement robust loading in `data_preprocessing.py`

#### Arsenal Match Statistics

- **Total Arsenal matches**: 919 matches
- **Home matches**: 443
- **Away matches**: 443
- **Results distribution**:
  - Wins: 48.1%
  - Losses: 29.3%
  - Draws: 22.6%
- **Goals**:
  - Average goals scored: 1.89 per match
  - Average goals conceded: 1.05 per match
- **Goal difference**: +747

#### Key Columns Identified

**Essential columns for modeling**:

- `Date`: Match date
- `HomeTeam`, `AwayTeam`: Team names
- `FTHG`, `FTAG`: Full-time goals (home/away)
- `FTR`: Full-time result (H/D/A)
- `HTHG`, `HTAG`: Half-time goals
- `HS`, `AS`: Shots (home/away)
- `HST`, `AST`: Shots on target
- `HF`, `AF`: Fouls
- `HC`, `AC`: Corners
- `HY`, `AY`: Yellow cards
- `HR`, `AR`: Red cards
- `B365H`, `B365D`, `B365A`: Betting odds (if available)

**Betting columns**: Many have missing values but can be useful features if available

---

## Project Enhancements

### Deep Learning Integration

- **Decision**: Added deep learning models to make project more research-oriented and distinctive
- **Models to implement**:
  1. Feedforward Neural Network (multi-layer perceptron)
  2. LSTM (Long Short-Term Memory) for sequence modeling
  3. Optional: Hybrid LSTM + Feature model
- **Rationale**:
  - Addresses professor's feedback about project being too similar to Kaggle projects
  - Adds research novelty with sequence modeling (LSTM)
  - Allows comparison: Traditional ML vs Deep Learning
- **Libraries added**: TensorFlow 2.20.0, Keras 3.12.0
- **Status**: Environment ready, models to be implemented in training phase

### Research Questions Formulated

1. Do deep learning models outperform traditional ML for EPL match prediction?
2. Does LSTM capture temporal patterns better than feedforward networks?
3. What sequence length (last N matches) works best for LSTM?
4. How do deep learning models handle class imbalance (Win/Draw/Loss)?
5. What features are most important for each model type?

---

## Next Steps

### Immediate (Data Preprocessing Phase)

- [ ] Create `src/data_preprocessing.py` with robust file loading
- [ ] Fix `epl-2004-05.csv` loading issue (encoding + parsing)
- [ ] Handle missing values (decide on strategy: drop, impute, or ignore)
- [ ] Standardize team names across all seasons
- [ ] Convert dates to datetime format
- [ ] Remove empty/unnamed columns
- [ ] Create unified cleaned dataset: `data/processed/epl_cleaned.csv`
- [ ] Filter and save Arsenal matches: `data/processed/arsenal_labeled.csv`

### Upcoming (Feature Engineering Phase)

- [ ] Design feature set (rolling averages, match context, etc.)
- [ ] Implement feature creation in `src/features.py`
- [ ] Ensure no data leakage (only use past information)
- [ ] Generate feature matrix: `data/processed/arsenal_features.csv`

### Future (Modeling Phase)

- [ ] Implement baseline models (Logistic Regression, Random Forest)
- [ ] Implement deep learning models (Feedforward NN, LSTM)
- [ ] Compare all models comprehensively
- [ ] Answer research questions with evidence

---

## Lessons Learned

1. **Data quality varies by season**: Different data sources mean different columns and formats
2. **Robust loading is essential**: Need error handling for encoding and parsing issues
3. **Exploration before preprocessing**: Understanding data issues first helps design better cleaning pipeline
4. **Deep learning adds complexity but also novelty**: Worth the effort for research-oriented project

---

## Notes

- Exploration notebook (`01_exploration.ipynb`) successfully identified all major data issues
- 25/26 files loading successfully is acceptable for now; will fix remaining in preprocessing
- Arsenal has good data coverage: 919 matches across 25 seasons
- Class distribution is somewhat imbalanced (48% wins, 29% losses, 23% draws) - may need to address in modeling
