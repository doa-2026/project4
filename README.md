# Predicting Income Level from Census Data

A machine learning project that predicts whether an individual earns **>$50K** or **≤$50K** per year using the UCI **Adult / Census Income** dataset.

## Dataset

- **Source:** `adult.csv` (UCI Adult / Census Income dataset)
- **Rows:** 48,842
- **Columns:** 15 (14 features + `income` target)
- **Target:** `income` — binary (`<=50K` = 0, `>50K` = 1)
- **Features:** age, workclass, fnlwgt, education, educational-num, marital-status, occupation, relationship, race, gender, capital-gain, capital-loss, hours-per-week, native-country

## Project Structure

The notebook is organized into three parts:

### Part 1 — EDA, Cleaning & Baseline Model
- Loaded data, checked dtypes, duplicates, and missing values
- Replaced `"?"` placeholders in `workclass`, `occupation`, and `native-country` with `"missing"`
- Explored relationships between features and income (capital-gain, education, age)
- Built a preprocessing pipeline: `SimpleImputer` + `StandardScaler` for numeric features, `SimpleImputer` + `OneHotEncoder` for categorical features
- Trained a baseline **Random Forest Classifier**
- Used **permutation importance** to identify the most predictive features

### Part 2 — Dimensionality Reduction, Clustering & Feature Selection
- Applied **PCA** (3 components) on top of the scaled features
- Ran **KMeans** clustering (k = 2–9), selected **k = 5** using silhouette score and the inertia elbow
- Added the cluster label as a new feature and retrained the Random Forest
- Applied **VarianceThreshold** and **SelectFromModel** to reduce the feature set
- Retrained and compared the model on the reduced feature set

### Part 3 — Neural Network & Hyperparameter Tuning
- Built a **Keras Sequential** neural network with `Dropout` and `EarlyStopping`
- Evaluated on the test set with a classification report and confusion matrix
- Tuned the network with **Keras Tuner (Hyperband)**, searching units, dropout, and optimizer
- Compared the tuned model against the untuned network and the Random Forest

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
keras-tuner
```

## How to Run

1. Place `adult.csv` in your working directory (the notebook originally loads it from Google Drive in Colab — update the `path` variable if running locally).
2. Run the notebook top to bottom — each part builds on objects created in the previous one (e.g., `X_train_scal`, `rf_pipe`).
3. Install Keras Tuner before Part 3: `pip install keras_tuner`

## Files

- `project4_.ipynb` — full analysis notebook
- `SUMMARY.md` — what was done and what was found
- `CONCLUSION.md` — final takeaways and recommendations
