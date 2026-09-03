# Bias-Variance Tradeoff and Regularization Techniques

## The Bias-Variance Tradeoff
The expected generalization error of a model can be decomposed mathematically into three terms:
$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$
- **Bias ($\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$)**: Error introduced by approximating a complex real-world relationship with a simplified model (Underfitting). High bias models miss relevant relations between features and targets.
- **Variance ($\text{Var}[\hat{f}(x)] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$)**: Sensitivity of the model to fluctuations in the training dataset (Overfitting). High variance models fit training noise rather than true underlying distribution.
- **Irreducible Error ($\sigma^2$)**: Inherent noise in the data/target mapping.

## Regularization Strategies
Regularization adds a penalty term $\Omega(w)$ to the loss function $\mathcal{L}(w)$ to constrain model complexity:
$$\mathcal{L}_{\text{reg}}(w) = \mathcal{L}(w) + \lambda \Omega(w)$$

1. **L1 Regularization (Lasso)**:
   - Penalty: $\Omega(w) = \sum_{j=1}^p |w_j|$.
   - Effect: Drives less important feature weights exactly to zero due to sharp diamond contours of the L1 ball intersecting loss contours on coordinate axes. Performs intrinsic feature selection.
2. **L2 Regularization (Ridge)**:
   - Penalty: $\Omega(w) = \sum_{j=1}^p w_j^2$.
   - Effect: Shrinks feature weights proportionally towards zero without making them exactly zero. Handles multicollinearity well by distributing weights across correlated features.
3. **ElasticNet**:
   - Convex combination of L1 and L2 penalties: $\lambda \left( \alpha \|w\|_1 + \frac{1 - \alpha}{2} \|w\|_2^2 \right)$.
   - Combines feature selection with group selection for correlated predictors.
4. **Early Stopping**:
   - Monitors validation loss during iterative optimization (gradient descent/boosting) and halts training when validation loss stops improving for $P$ patience epochs.
