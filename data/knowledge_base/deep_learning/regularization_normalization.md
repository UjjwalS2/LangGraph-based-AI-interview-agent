# Regularization and Normalization in Deep Learning

## Dropout
- Randomly sets neuron activations to zero with probability $p$ during training:
  $$\tilde{a} = a \odot m, \quad m \sim \text{Bernoulli}(1 - p)$$
- **Inverted Dropout**: Scales active activations by $\frac{1}{1 - p}$ during training so no scaling is required at test/inference time.
- Prevents co-adaptation of feature detectors, forcing network to learn redundant representations across sub-networks.

## Batch Normalization (BatchNorm)
Normalizes layer inputs across the mini-batch dimension:
$$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$
- Learnable scale $\gamma$ and shift $\beta$ preserve expressive capacity.
- Maintains running exponential moving averages $(\mu_{\text{running}}, \sigma_{\text{running}}^2)$ for deterministic inference.
- Reduces internal covariate shift, smooths the loss optimization landscape, and acts as mild regularizer.
- Limitations: Ineffective for small batch sizes ($m < 8$) and variable-length sequences.

## Layer Normalization (LayerNorm)
Normalizes across features/channels for each individual training sample independently:
$$\mu_L = \frac{1}{H} \sum_{i=1}^H x_i, \quad \sigma_L^2 = \frac{1}{H} \sum_{i=1}^H (x_i - \mu_L)^2$$
- Independent of batch size, making it ideal for RNNs and Transformers.
- **RMSNorm (Root Mean Square Normalization)**: Omits mean-centering $(\mu=0)$ and only scales by root mean square, reducing compute overhead with equivalent empirical performance (used in LLaMA, Gemma).
