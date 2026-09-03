# Feature Engineering, Encoding, and Data Transformation

## Missing Value Imputation
- **Univariate Imputation**: Mean/median for numerical variables; mode or constant `__MISSING__` category for categoricals. (Median is robust against extreme outliers).
- **Multivariate Imputation**: KNN Imputation (distance-weighted averaging of $k$ nearest instances) or MICE (Multivariate Imputation by Chained Equations).
- **Missingness Indicators**: Adding a binary column `is_missing_feature` allows models to learn informatively missing patterns.

## Categorical Encoding
- **One-Hot Encoding**: Creates $k$ binary columns for $k$ categories (or $k-1$ to avoid dummy variable trap in linear models). Inefficient for high cardinality (curse of dimensionality).
- **Ordinal Encoding**: Maps categories to integers when natural ordering exists (e.g. Low=1, Medium=2, High=3).
- **Target Encoding (Mean Encoding)**: Replaces category with mean target value for that category.
  - Risk: Severe target leakage and overfitting.
  - Mitigation: Smoothing with global prior $S = \frac{n \cdot \bar{y}_c + m \cdot \bar{y}_{global}}{n + m}$, additive Gaussian noise, and Out-of-Fold (OOF) computation.
- **Binary & Hashing Encoding**: Maps categories to binary representations or uses hashing trick (`hash(category) % num_buckets`) to constrain memory.

## Feature Scaling and Distribution Transformation
- **Min-Max Normalization**: $x' = \frac{x - x_{min}}{x_{max} - x_{min}} \in [0, 1]$. Sensitive to outliers.
- **Standardization (Z-Score)**: $z = \frac{x - \mu}{\sigma}$, yielding zero mean and unit variance. Suitable for gradient-based models and distance metrics (SVM, KNN, PCA).
- **Power Transformations**:
  - Log Transform: $y = \log(x + 1)$ stabilizes right-skewed positive data.
  - Box-Cox & Yeo-Johnson: Parametric transformations finding optimal parameter $\lambda$ to make distribution approximately Gaussian.

## Feature Selection
- **Filter Methods**: Pearson correlation, Mutual Information, ANOVA F-test.
- **Wrapper Methods**: Recursive Feature Elimination (RFE), Sequential Forward Selection.
- **Embedded Methods**: Lasso coefficients, Tree-based Mean Decrease in Impurity (MDI), and Permutation Feature Importance.
