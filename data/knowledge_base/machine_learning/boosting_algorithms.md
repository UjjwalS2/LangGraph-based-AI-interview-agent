# Gradient Boosting, XGBoost, and LightGBM

## Boosting Foundations
- **AdaBoost (Adaptive Boosting)**: Sequentially trains base classifiers. After each round, misclassified samples are assigned higher weights $w_i \leftarrow w_i \exp(\alpha_t)$ and correctly classified samples lower weights. Predictions are aggregated via weighted majority vote.
- **Gradient Boosting Machine (GBM)**: Formulates boosting as gradient descent in function space. At iteration $t$, the new tree $h_t(x)$ is fit directly to the pseudo-residuals (negative gradient of the loss function $\mathcal{L}(y, f_{t-1}(x))$):
  $$r_{it} = -\left[ \frac{\partial \mathcal{L}(y_i, f(x_i))}{\partial f(x_i)} \right]_{f(x) = f_{t-1}(x)}$$
  Model update: $f_t(x) = f_{t-1}(x) + \eta h_t(x)$, where $\eta \in (0, 1]$ is the shrinkage learning rate.

## XGBoost (Extreme Gradient Boosting)
XGBoost introduces several mathematical and algorithmic optimizations:
1. **Second-Order Taylor Expansion**:
   Approximates the loss function using both the first derivative (gradient $g_i$) and second derivative (hessian $h_i$):
   $$\tilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
   where $g_i = \partial_{\hat{y}^{(t-1)}} \mathcal{L}(y_i, \hat{y}^{(t-1)})$ and $h_i = \partial^2_{\hat{y}^{(t-1)}} \mathcal{L}(y_i, \hat{y}^{(t-1)})$.
2. **Explicit Regularization $\Omega(f)$**:
   $$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2 + \alpha \sum_{j=1}^T |w_j|$$
   - $T$: Number of terminal leaf nodes.
   - $w_j$: Output score/weight of leaf $j$.
   - **$\gamma$ (gamma)**: Minimum loss reduction required to make a further partition on a leaf node. Serves as tree-pruning complexity parameter.
   - $\lambda$ (reg_lambda): L2 regularization on leaf weights.
   - $\alpha$ (reg_alpha): L1 regularization on leaf weights.
3. **Optimal Leaf Weight and Split Gain**:
   - Optimal weight for leaf $j$: $w_j^* = -\frac{G_j}{H_j + \lambda}$, where $G_j = \sum_{i \in I_j} g_i$ and $H_j = \sum_{i \in I_j} h_i$.
   - Split Gain Formula:
     $$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
   - If $\text{Gain} < 0$ (or $<\gamma$), the branch is pruned.
4. **Column and Row Subsampling**: `colsample_bytree`, `subsample` prevent overfitting.
5. **Weighted Quantile Sketch & Sparsity-Aware Split Finding**: Handles non-uniform sample distributions and natively routes missing values to the default branch that maximizes gain.

## LightGBM (Light Gradient Boosting Machine)
LightGBM addresses memory and computational bottlenecks of gradient boosting:
1. **Histogram-based Algorithm**: Buckets continuous feature values into discrete bins (e.g. 256 bins), reducing memory consumption by $8\times$ and speeding up split evaluation to $O(\text{#bins})$.
2. **GOSS (Gradient-based One-Side Sampling)**: Keeps all instances with large gradients (high error) and randomly samples a subset $b$ of instances with small gradients, scaling small-gradient samples by $\frac{1-a}{b}$ to preserve original data distribution.
3. **EFB (Exclusive Feature Bundling)**: Bundles mutually exclusive sparse features (which rarely take non-zero values simultaneously) into a single dense feature.
4. **Leaf-wise (Best-First) Tree Growth**: Expands the leaf that maximizes loss reduction rather than level-wise (depth-first) growth, achieving lower loss at the risk of overfitting if `max_depth` or `num_leaves` is unconstrained.
