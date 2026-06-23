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

## 🧰 Tools Used

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

-Predict income category (>50K or <=50K).

-Build a baseline classification model.

-Perform exploratory data analysis (EDA).

-Apply feature engineering and preprocessing techniques.

-Evaluate feature selection and dimensionality reduction methods.

-Explore clustering as an additional feature-generation approach.

-Develop and evaluate a neural network model.

-Compare model performance using multiple evaluation metrics.


## ⚙️ Project Workflow

### Part 1: Data Understanding and Exploration
-Load and inspect the dataset.

-Check data types and missing values.

-Generate descriptive statistics.

--Visualize feature distributions.

Analyze relationships between features and the target variable.

### Part 2: Data Preprocessing

-Handle missing values.

-Encode categorical variables.

-Scale numerical features.

-Split the data into training and testing sets.

### Part 3: Baseline Model
-Train a Random Forest Classifier using all available features.

-Evaluate model performance using:

--Accuracy

--Precision

--Recall

--F1-score

### Part 4: Feature Selection

-Apply Filter Methods.

-Apply Backward Wrapper Selection.

-Compare results with the baseline model.

### Part 5: Dimensionality Reduction

-Apply PCA (Principal Component Analysis).

-Train models using reduced feature sets.

-Compare performance with previous approaches.

### Part 6: Clustering

-Apply clustering techniques.

-Use cluster assignments as additional features.

-Evaluate their impact on classification performance.

### Part 7: Neural Network Model

-Build a Sequential Neural Network.

-Apply Early Stopping to reduce overfitting.

-Apply hyperparamter tunning to improve the performance of the neural network.

-Compare neural network performance with traditional machine learning models.

## 📈 Results

-The baseline Random Forest model achieved strong classification performance. 

-Feature selection reduced the number of features while maintaining competitive results.

-PCA reduced dimensionality with only a minor impact on performance.

-Clustering provided additional information that was evaluated as part of the modeling process.

-The neural network achieved approximately 86% accuracy with Early Stopping and produced results comparable to other advanced approaches.


  ## 📊 Conclusion

This project developed machine learning models to predict whether an individual's income is above $50K using demographic and employment data. Data preprocessing was performed, including handling missing values, encoding categorical features, scaling, and data visualization. Different approaches such as feature selection, dimensionality reduction, clustering, and neural networks were applied and compared. The results showed that proper preprocessing and feature engineering improve model performance and help identify important factors affecting income. Overall, the project provided a good understanding of classification modeling and income prediction.



