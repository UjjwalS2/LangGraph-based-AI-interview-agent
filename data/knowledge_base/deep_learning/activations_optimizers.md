# Activation Functions and Gradient-Based Optimizers

## Activation Functions
Non-linear activation functions allow neural networks to approximate arbitrary continuous functions (Universal Approximation Theorem):
- **Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$. Non-zero centered, saturates at tails causing vanishing gradients.
- **Tanh**: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1, 1)$. Zero-centered, but still suffers from saturation in deep models.
- **ReLU (Rectified Linear Unit)**: $f(z) = \max(0, z)$. Fast to compute, derivative is 1 for $z > 0$ (prevents gradient vanishing), but susceptible to "Dying ReLU" when neurons receive negative inputs permanently.
- **LeakyReLU & PReLU**: $f(z) = \max(\alpha z, z)$ with small slope $\alpha$ for $z < 0$, avoiding dying neurons.
- **GELU (Gaussian Error Linear Unit)**: $f(x) = x \cdot \Phi(x) = x P(X \le x), X \sim \mathcal{N}(0, 1)$. Smooth non-monotonic function used standardly in modern transformers (BERT, GPT, RoBERTa).
- **Swish / SiLU**: $f(x) = x \cdot \sigma(\beta x)$. Discovered by neural architecture search; provides smooth gradient landscape.
- **Softmax**: $\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$. Converts raw logits into a normalized probability distribution over $K$ classes.

## First and Second-Moment Optimizers
1. **Stochastic Gradient Descent with Momentum**:
   Accumulates velocity vector in directions of persistent gradients, dampening oscillations:
   $$v_t = \gamma v_{t-1} + \eta \nabla_\theta \mathcal{L}(\theta)$$
   $$\theta_{t+1} = \theta_t - v_t$$
2. **RMSprop**:
   Maintains an exponentially decaying moving average of squared gradients to scale learning rate per parameter:
   $$s_t = \beta s_{t-1} + (1 - \beta) g_t^2$$
   $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t$$
3. **Adam (Adaptive Moment Estimation)**:
   Combines first moment $m_t$ (mean of gradients) and second moment $v_t$ (uncentered variance of gradients) with bias corrections:
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
   $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
4. **AdamW (Decoupled Weight Decay)**:
   Standard Adam with L2 regularization incorporates weight decay directly into gradient updates, causing the effective regularization to be scaled down for weights with large gradients. AdamW explicitly decouples weight decay from the gradient step:
   $$\theta_{t+1} = \theta_t - \eta \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
   Standard optimizer in modern Transformer and LLM pre-training.
