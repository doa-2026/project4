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

## Tools Used

**Programming Language:**
Python

**Libraries:**
pandas – data manipulation and preprocessing
numpy – numerical computations
scikit-learn – machine learning models and preprocessing pipelines
matplotlib / seaborn – data visualization

**Machine Learning Techniques:**

Random Forest Classifier (baseline model)
OneHot Encoding & Standard Scaling
SimpleImputer for missing values
Permutation Importance for feature analysis
Feature Selection methods
Dimensionality Reduction (e.g., PCA)
Clustering techniques
Neural Networks (experimentation phase)

## Project Structure

### Part 1 — Cleaning & Baseline Model
- Loaded data, checked dtypes, duplicates, and missing values.
- Checked for and addressed inconsistent or placeholder values in categorical features.
- Explored relationships between features and target.
- Built a preprocessing pipeline to handle missing values and encode/scale features appropriately for modeling.
- Trained a baseline Random Forest Classifier.
- Evaluated feature importance using permutation importance to determine the most influential predictors.

### Part 2 — Dimensionality Reduction, Clustering & Feature Selection
- Applied PCA to the scaled feature set to reduce dimensionality
- Performed KMeans clustering and selected the optimal number of clusters using silhouette score and the elbow method (inertia).
- Used cluster assignments as additional features and retrained the Random Forest model.
- Applied feature selection techniques, including VarianceThreshold and SelectFromModel, to reduce the feature space.
- Retrained and evaluated the model using the reduced feature set for performance comparison.

### Part 3 — Neural Network & Hyperparameter Tuning
- Built a small Keras Sequential neural network with  one hidden layer and trained it over multiple epochs, then improved performance using regularization techniques(Early stopping & Dropout).
- Evaluated on the test set with a classification report and confusion matrix.
- Tuned the neural network using Keras Tuner (Hyperband) by optimizing the number of units, dropout rate, and optimizer.
- Compared the performance of the tuned neural network against the baseline (untuned) model and the Random Forest classifier.


A Random Forest Classifier was trained on the full preprocessed feature set as a baseline model for income prediction.


   --------------------------------------------------------------------
 Classification Metrics: Training Data
----------------------------------------------------------------------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     27841
           1       1.00      1.00      1.00      8751

    accuracy                           1.00     36592
   macro avg       1.00      1.00      1.00     36592
weighted avg       1.00      1.00      1.00     36592



Classification Metrics: Test Data
----------------------------------------------------------------------
              precision    recall  f1-score   support

           0       0.89      0.93      0.91      9268
           1       0.74      0.62      0.68      2930

    accuracy                           0.86     12198
   macro avg       0.82      0.78      0.79     12198
weighted avg       0.85      0.86      0.85     12198
