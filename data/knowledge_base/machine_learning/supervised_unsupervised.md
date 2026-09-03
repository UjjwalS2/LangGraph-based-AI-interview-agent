# Supervised and Unsupervised Machine Learning Paradigms

## Supervised Learning
Learning a mapping function $f: X \to Y$ given labelled pairs $(x_i, y_i)$:
- **Classification**: Target variable $y$ is discrete/categorical (binary, multiclass, multilabel). Examples: Logistic Regression, Support Vector Machines (SVM), Random Forests, Neural Networks.
- **Regression**: Target variable $y$ is continuous ($y \in \mathbb{R}$). Examples: Linear Regression, Ridge/Lasso, Gradient Boosted Trees, SVR.

## Unsupervised Learning
Finding inherent structure, patterns, or probability densities $P(X)$ in unlabelled data:
1. **Clustering**:
   - **K-Means**: Partitioning algorithm minimizing inertia (Within-Cluster Sum of Squares, WCSS). Assumes spherical, equal-variance clusters. Sensitive to initial centroid placement (mitigated by K-Means++).
   - **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Discovers arbitrarily shaped clusters based on core points, $\epsilon$-neighborhood, and `min_samples`. Identifies outliers as noise.
   - **Hierarchical Clustering**: Agglomerative (bottom-up) or Divisive (top-down) clustering using linkage criteria (Ward, Complete, Single, Average).
2. **Dimensionality Reduction**:
   - **Principal Component Analysis (PCA)**: Linear transformation finding orthogonal axes (eigenvectors of covariance matrix) that maximize variance.
   - **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Non-linear manifold technique modeling pairwise similarities as probabilities, primarily for 2D/3D visualization.
   - **UMAP (Uniform Manifold Approximation and Projection)**: Preserves global and local topological structure with faster scalability than t-SNE.

## Semi-Supervised and Self-Supervised Learning
- **Semi-Supervised**: Uses a small set of labelled data with a large unlabelled dataset (e.g. Pseudo-labelling, label propagation).
- **Self-Supervised**: Generates supervisory signals directly from data (e.g. Masked Language Modeling in BERT, Contrastive Learning in SimCLR/CLIP).
