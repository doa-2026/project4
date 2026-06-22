# Machine Learning Model for Predicting Income Level

A supervised machine learning project that classifies individuals’ income levels as either above or below $50K per year based on demographic and employment features from the UCI Adult (Census Income) dataset.

## Dataset

- **Source:** `adult.csv`
- **Rows:** 48,842 
- **Columns:** 15 (14 features + `income` target)
- **Target:** `income` — binary (`<=50K` = 0, `>50K` = 1)
- **Features:** age, workclass, fnlwgt, education, educational-num, marital-status, occupation, relationship, race, gender, capital-gain, capital-loss, hours-per-week, native-country

## Project Summary

Developed a classification model to predict income level (>50K vs ≤50K) using the UCI Census dataset, based on 14 demographic and employment features.

The project evaluates multiple machine learning strategies beyond a baseline model, including feature selection, dimensionality reduction, clustering, and neural networks, to assess their impact on performance and interpretability.

Focus was placed on improving generalization and understanding key drivers of income prediction through exploratory analysis and feature importance techniques.

Tools Used

Programming Language:
Python

Libraries:

pandas – data manipulation and preprocessing
numpy – numerical computations
scikit-learn – machine learning models and preprocessing pipelines
matplotlib / seaborn – data visualization

Machine Learning Techniques:

Random Forest Classifier (baseline model)
OneHot Encoding & Standard Scaling
SimpleImputer for missing values
Permutation Importance for feature analysis
Feature Selection methods
Dimensionality Reduction (e.g., PCA)
Clustering techniques
Neural Networks (experimentation phase)


### Part 1 — EDA, Cleaning & Baseline Model
- Loaded data, checked dtypes, duplicates, and missing values.
- Checked for and addressed inconsistent or placeholder values in categorical features.
- Explored relationships between features and target.
- Built a preprocessing pipeline to handle missing values and encode/scale features appropriately for modeling.
- Trained a baseline Random Forest Classifier.
- Evaluated feature importance using permutation importance to determine the most influential predictors.

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





- `CONCLUSION.md` — final takeaways and recommendations
