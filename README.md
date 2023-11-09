# Predictive-Modeling-and-Evaluation-in-Machine-Learning
CS 484: Introduction to Machine Learning Assignment

# CS 484: Introduction to Machine Learning - Spring 2023
## Assignment 4

This assignment involves a series of questions related to machine learning algorithms and their applications in classification and prediction tasks.

---

## Question 1: Analysis of Classification Tree (30 Points)

A classification tree diagram has been provided for the label variable Payment Type, which can be either Cash or Credit Card.

### Part (a) - Area Under Curve (10 points)

Describe the steps and calculate the Area Under Curve (AUC) value for the provided classification tree.

### Part (b) - Root Average Squared Error (10 points)

Explain the steps to calculate the Root Average Squared Error (RASE) for the classification tree model.

### Part (c) - Model Acceptability (10 points)

Evaluate the acceptability of the model based on the AUC and RASE metrics.

---

## Question 2: Classification Model Analysis (30 points)

A classification model has been trained, and a table of predicted event probabilities for twenty observations is provided.

### Part (a) - Kolmogorov–Smirnov Curve (10 points)

Generate the Kolmogorov–Smirnov curve, identify, and provide the probability threshold yielding the highest Kolmogorov–Smirnov statistic.

### Part (b) - Precision-Recall Curve (10 points)

Generate the Precision-Recall curve, determine, and provide the probability threshold yielding the highest F1 Score.

### Part (c) - Misclassification Rates (10 points)

Calculate the misclassification rates using the thresholds determined in parts (a) and (b).

---

## Question 3: Homeowner Claim History Analysis (40 points)

The `Homeowner_Claim_History.xlsx` file contains data on homeowner policy claims. The task is to predict the natural logarithm of the Severity, which is the average claim amount per year, using seven categorical predictors.

### Part (a) - Multi-Layer Perceptron Training (20 points)

Train a Multi-Layer Perceptron neural network using grid search to select the optimal network structure. Document the search results in a detailed table showing the relevant metrics and statistics.

### Part (b) - Optimal Network Structure (10 points)

Identify the network structure with the lowest root mean squared error on the testing partition from the networks that converged.

### Part (c) - Severity Predictions (10 points)

Analyze the categorical predictor combinations to determine which yield the lowest and highest Severity.


