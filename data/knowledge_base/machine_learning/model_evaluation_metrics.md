# Model Evaluation Metrics and Validation Strategies

## Classification Metrics
- **Confusion Matrix**: True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).
- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$. Proportion of positive identifications that were actually correct. Crucial when False Positives are costly (e.g. spam filtering).
- **Recall (Sensitivity / True Positive Rate)**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$. Proportion of actual positives identified correctly. Crucial when False Negatives are dangerous (e.g. cancer diagnosis, fraud detection).
- **F1-Score**: Harmonic mean of Precision and Recall:
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$
- **Multiclass Averaging**:
  - **Macro Average**: Unweighted mean of metric across all classes (treats all classes equally, sensitive to rare classes).
  - **Micro Average**: Global aggregate computation over total TP, FP, FN (dominated by majority classes).
  - **Weighted Average**: Averages metrics weighted by class support (number of true instances).
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**:
  - Plots True Positive Rate ($y$-axis) vs False Positive Rate ($x$-axis) across all classification thresholds.
  - Represents the probability that the classifier ranks a randomly chosen positive instance higher than a randomly chosen negative instance.
  - Insensitive to class imbalance (can be overly optimistic when negatives vastly outnumber positives).
- **PR-AUC (Precision-Recall Area Under Curve)**:
  - Better metric than ROC-AUC for severe class imbalance because it does not involve True Negatives in its formulation.

## Cross-Validation Strategies
- **K-Fold Cross-Validation**: Divides data into $K$ equal folds; iteratively trains on $K-1$ and tests on the remaining fold.
- **Stratified K-Fold**: Preserves the percentage of samples for each class in every fold. Essential for imbalanced classification.
- **TimeSeriesSplit / Purged CV**: Prevents data leakage and lookahead bias in chronological data by using expanding training windows ($t_1 \dots t_k$) to predict future window ($t_{k+1}$).
