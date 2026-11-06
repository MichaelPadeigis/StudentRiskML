# Results

Data: UCI Predict Students Dropout and Academic Success (4,424 rows).
Outcome: Dropout vs Graduate, excluding unresolved Enrolled students.
80/20 stratified train/test split, seed 42, five-fold cross-validation within training.
All test results are for the selected model evaluated once on the held-out test set.

| Prediction time | Chosen model | Features | Test ROC-AUC | Test recall | Test precision | Test F1 | Test accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| enrollment | logistic_regression | 24 | 0.855 | 0.775 | 0.673 | 0.720 | 0.764 |
| after first semester | svm | 30 | 0.956 | 0.887 | 0.890 | 0.889 | 0.913 |

The cutoff for predicting dropout is 0.50. Risk labels are display categories using
illustrative 0.25 / 0.60 cutoffs, not proven thresholds for offering student support.
This prototype is not suitable for real admissions, academic, or intervention decisions.

Read each experiment folder for the 4-model CV comparison, test plots, and predictions.
