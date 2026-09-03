# Decision Trees, Bagging, and Random Forests

## Decision Tree Fundamentals
Decision trees recursively partition feature space into axis-aligned rectangular regions:
- **Splitting Criteria**:
  - **Gini Impurity (CART)**: $I_G(p) = 1 - \sum_{i=1}^C p_i^2$. Measures the probability that a randomly chosen element from the set would be incorrectly labeled.
  - **Information Gain / Entropy (ID3, C4.5)**: $H(p) = -\sum_{i=1}^C p_i \log_2(p_i)$. Information Gain is reduction in entropy after splitting.
  - **Mean Squared Error (MSE) / Variance Reduction**: For regression trees.
- **Tree Pruning**:
  - Pre-pruning (Early Stopping): `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`.
  - Post-pruning (Cost-Complexity Pruning): Minimizes $R_\alpha(T) = R(T) + \alpha |T|$, where $|T|$ is number of terminal leaves.

## Ensemble Paradigms: Bagging vs. Boosting
- **Bagging (Bootstrap Aggregating)**: Builds independent base learners in parallel on distinct bootstrap samples and averages their predictions. **Primarily reduces variance** without significantly increasing bias.
- **Boosting**: Builds base learners sequentially, where each new learner focuses on errors/residuals of the existing ensemble. **Primarily reduces bias**.

## Random Forest Architecture
A Random Forest improves upon standard bagging of decision trees through double randomization:
1. **Bootstrap Sampling**: Each tree is trained on an independently drawn bootstrap sample ($N$ instances with replacement from original dataset, leaving $\approx 36.8\%$ as Out-of-Bag (OOB) samples).
2. **Feature Subsampling (Random Subspace Method)**: At every candidate split point in every tree, only a random subset of features $m \le p$ is considered (typically $m = \sqrt{p}$ for classification, $m = p/3$ for regression).

## Why Random Forest Reduces Variance
The variance of the average of $B$ identically distributed trees, each with individual variance $\sigma^2$ and pairwise correlation $\rho$, is:
$$\text{Var}(\bar{T}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
- As $B \to \infty$, the second term $\to 0$, but the lower bound is dictated by $\rho \sigma^2$.
- By subsampling random features at every split, Random Forest de-correlates the individual trees (drastically reducing pairwise correlation $\rho$).
- Consequently, total ensemble variance drops dramatically compared to both an individual deep decision tree and standard bagged trees.

## Out-of-Bag (OOB) Evaluation
Because each bootstrap sample leaves roughly $1/e \approx 36.8\%$ of instances unused by that specific tree, these OOB instances serve as a built-in cross-validation set. The OOB error provides an unbiased estimate of the test set generalization error without requiring a separate validation split.
