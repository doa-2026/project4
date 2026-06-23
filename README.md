# Machine Learning Model for Predicting Income Level

## 📌 Overview

A supervised machine learning project that classifies individuals’ income levels as either above or below $50K per year based on demographic and employment features from the UCI Adult (Census Income) dataset.

## 📂 Dataset

- **Source:** `adult.csv`
- **Rows:** 48,842 
- **Columns:** 15 (14 features + `income` target)
- **Target:** `income` — binary (`<=50K` = 0, `>50K` = 1)
- **Features:** age, workclass, fnlwgt, education, educational-num, marital-status, occupation, relationship, race, gender, capital-gain, capital-loss, hours-per-week, native-country

## 📌 Project Summary

Developed a machine learning project to predict whether an individual earns more than $50K or ≤$50K per year using the UCI Census dataset, which includes 14 demographic and employment-related features such as age, education, occupation, and capital gain. The project began with a baseline classification model and then extended to a broader investigation of multiple machine learning strategies, including feature selection, dimensionality reduction, clustering techniques, and neural networks. Each approach was evaluated to understand its effect on model performance, generalization ability, and interpretability, with the goal of identifying the most effective methods for improving predictive accuracy and gaining deeper insight into the factors influencing income levels.

## 📄 Tools Used

**Programming Language:**
Python

**Libraries:**
pandas – data manipulation and preprocessing
numpy – numerical computations
scikit-learn – machine learning models and preprocessing pipelines
matplotlib / seaborn – data visualization
TensorFlow / Keras-Neural Network model building

**Machine Learning Techniques:**

Random Forest Classifier (baseline model)
OneHot Encoding & Standard Scaling
SimpleImputer for missing values
Permutation Importance for feature analysis
Feature Selection methods
Dimensionality Reduction (e.g., PCA)
Clustering techniques
Neural Networks

## 🎯 Objectives

The main objectives of this project are:

-To predict income level (>50K or ≤50K) using demographic and employment data.

-To build and evaluate a baseline machine learning model using Random Forest.

-To investigate the impact of feature selection on model performance and efficiency.

-To apply PCA for dimensionality reduction and compare results with the original feature space.

-To use clustering techniques to discover hidden structures in the dataset.

-To develop a neural network model and optimize its hyperparameters for better accuracy.

-To compare multiple models and identify the best-performing approach.

-To demonstrate a complete machine learning workflow from preprocessing to evaluation.

## ⚙️ Workflow

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

##  📊 Project Details & Steps 
1️⃣ Data Understanding , Cleanning & visulization 
🎯 Goal:
Predict whether a person earns:
≤50K
>50K
🧠 Type of Problem:
Binary classification problem

🔍 Steps:
- Check missing & inconsistent values
- Understand feature distributions
- Explored relationships between features and target

 Data Preprocessing
 

5️⃣ Baseline Model
🌲 Model used:
Random Forest Classifier
⚙️ Steps:
Train model on full dataset
Predict test data
Evaluate performance:
Accuracy
Precision
Recall
F1-score
🎯 Purpose:

To set a reference performance

6️⃣ Feature Selection
🎯 Goal:

Reduce unnecessary features

⚙️ Steps:
Use feature importance (Random Forest)
Remove low-impact features
Retrain model
📈 Benefit:
Faster training
Less noise
Better generalization
7️⃣ Dimensionality Reduction (PCA)
🎯 Goal:

Reduce number of features while keeping information

⚙️ Steps:
Standardize data
Apply PCA
Choose number of components
Train model on PCA data
📊 Comparison:
Original features vs reduced features
8️⃣ Clustering Analysis
🎯 Goal:

Discover hidden groups in data

⚙️ Steps:
Apply K-Means clustering
Choose optimal number of clusters (Elbow method)
Visualize clusters
🧠 Insight:

Groups people with similar income behavior patterns

9️⃣ Neural Network Model
🤖 Model:

Deep Learning (Keras / TensorFlow)

⚙️ Steps:
Build sequential model
Add hidden layers (ReLU activation)
Output layer (Sigmoid)
Compile model (Adam optimizer)
🛑 Prevent overfitting:
Dropout layers
Early stopping
Hyperparameter tuning
🔟 Model Evaluation & Comparison
📊 Compare models:
Random Forest (baseline)
Feature selection model
PCA model
Neural network
📈 Metrics used:
Accuracy
F1-score
Generalization performance
1️⃣1️⃣ Final Result
🏆 Select best model based on:
Highest accuracy
Stable performance on test data
Low overfitting
1️⃣2️⃣ Conclusion
📌 Summarize:
Best performing method
Impact of feature selection
Effect of PCA
Neural network performance
🧠 Full Pipeline Summary (Simple View)
Data → Clean → Explore → Train Baseline → Improve Features → PCA → Clustering → Neural Network → Compare → Final Model

If you want, I can also:

turn this into a presentation (PowerPoint slides)
make it a GitHub README full report
or 
draw a workflow diagram (very useful for interviews)



