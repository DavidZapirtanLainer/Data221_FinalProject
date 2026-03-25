# Predicting Heart Failure: A Comparative Analysis of Supervised Learning Models

DATA 221: Introduction to Data Science — University of Calgary
Ahmed, Fahim, Abdullah, David

## Project Overview

This project builds and compares six supervised machine learning models to predict the presence of heart disease using the Heart Failure Prediction dataset from Kaggle. The dataset contains 918 patient records with clinical and demographic features such as age, cholesterol, resting blood pressure, and exercise-induced symptoms. Models are evaluated on accuracy, precision, recall, F1-score, and ROC-AUC, with a priority on recall to minimize missed heart disease cases.

## Dataset

Download heart.csv from the Kaggle Heart Failure Prediction dataset and place it in the root of this repository before running any notebooks.
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

## Files

**heart.csv**
The raw dataset containing 918 patient records and 11 clinical features. This file must be present in the same directory as the notebooks for any of them to run.

**data_sort_and_split.ipynb**
Handles all data cleaning, encoding, and train-test splitting that is shared across every model. Categorical string columns are encoded to numeric values and the dataset is split 80/20 with a fixed random seed. Each model notebook replicates this preprocessing block at the top since notebooks run independently.

**Logistic_Regression_Model.ipynb**
Trains a Logistic Regression classifier as the interpretable baseline model for the project. Features are scaled using StandardScaler fitted on training data only, and the model is evaluated using all five metrics plus a confusion matrix.

**KNN_Model.ipynb**
Trains a K-Nearest Neighbors classifier, testing values of k from 1 to 20 to find the best performing k before rebuilding the final model. Features are scaled since KNN relies on distance calculations and is sensitive to differences in feature ranges.

**Decision_Tree_Model.ipynb**
Trains a Decision Tree classifier with a max depth of 5 and entropy as the splitting criterion to balance performance and avoid overfitting. No scaling is applied since decision trees are not distance-based. A feature importance chart is included to show which patient features the model relied on most.

**Random_Forest_Model.ipynb**
Trains a Random Forest classifier using 100 decision trees with entropy splitting and a max depth of 10. No scaling is required. A feature importance chart is included showing the combined contribution of each feature across all trees.

**Support_Vector_Machine.ipynb**
Trains a Support Vector Machine with a linear kernel and standard scaling applied to the features. The probability flag is enabled so ROC-AUC can be calculated alongside the other metrics.

**Neural_Network.ipynb**
Trains a single hidden layer neural network using TensorFlow/Keras with early stopping to prevent overfitting on the small dataset. Features are scaled before training and the sigmoid output is thresholded at 0.5 to produce final predictions.

**Comparison_Metrics.ipynb**
Aggregates the final metric results from all six models and produces bar charts comparing them side by side. The metrics are entered manually from each model notebook to avoid re-running the full training pipelines.

## How to Run

Run data_sort_and_split.ipynb first to verify the dataset loads and encodes correctly. Then run each model notebook independently in any order. Run Comparison_Metrics.ipynb last after all model results have been recorded.

## Dependencies

- Python 3
- pandas, numpy, scikit-learn, matplotlib
- tensorflow (Neural Network notebook only)
