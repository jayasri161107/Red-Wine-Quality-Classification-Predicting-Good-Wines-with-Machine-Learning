Red Wine Quality Prediction Using Machine Learning
A supervised machine learning project that predicts whether a red wine is good quality from its physicochemical properties, using Python and scikit-learn.

Project Overview
Wine quality is traditionally judged by human experts, which is slow and subjective. This project uses laboratory measurements (acidity, sulphates, alcohol, etc.) to build classification models that predict wine quality automatically.

Objective: Build and compare machine learning models that classify a red wine as good or not good.

Target definition: The original quality score (3–8) is converted into a binary label:

Label	Meaning	Rule
1	Good wine	quality ≥ 7
0	Not good	quality ≤ 6

Dataset
Source: Red Wine Quality dataset (UCI Machine Learning Repository / Kaggle)
File: winequality-red.csv (semicolon-separated)
Size: 1,599 samples, 11 input features + 1 quality score
Missing values: None
Duplicates: 240 duplicate rows removed (1,359 remain)
Class balance: ~13.6% good wines, so the data is imbalanced


Project Workflow
Import libraries and load the data
Statistical analysis (info, describe, target distribution)
Missing value and duplicate handling
Outlier treatment using IQR capping, with boxplots before and after
Feature/target separation and column cleaning
Data visualisation (distributions, group comparisons, correlation heatmap)
Stratified 80/20 train–test split
Feature selection with SelectKBest (ANOVA F-test)
Feature scaling with StandardScaler (fit on training data only, to avoid data leakage)
Model training and evaluation
Model comparison and conclusion

Models Used
Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors (KNN)

Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC and confusion matrices.

Results
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression: Accuracy 0.779, Precision 0.374, Recall 0.919, F1-Score 0.531, ROC-AUC 0.890
Random Forest: Accuracy 0.868, Precision 0.513, Recall 0.541, F1-Score 0.526, ROC-AUC 0.883
Decision Tree: Accuracy 0.790, Precision 0.361, Recall 0.703, F1-Score 0.477, ROC-AUC 0.789
KNN: Accuracy 0.868, Precision 0.517, Recall 0.405, F1-Score 0.455, ROC-AUC 0.800

Key Findings
Most influential features: alcohol, volatile acidity, sulphates and citric acid. Residual sugar and pH matter least.
Logistic Regression gives the best F1 and ROC-AUC and identifies about 92% of good wines, but produces more false positives.
Random Forest is the most balanced model (highest accuracy together with reasonable precision and recall).
Because only ~14% of wines are "good", accuracy alone is misleading, so precision, recall and F1 are used.

Technologies
Python 3
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
Jupyter Notebook

Author
Jayasri C S
B.Sc. Computer Science with Artificial Intelligence 
M.O.P. Vaishnav College for Women, Chennai
