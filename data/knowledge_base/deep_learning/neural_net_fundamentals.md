# Neural Network Fundamentals and Backpropagation

## Multilayer Perceptron (MLP) Architecture
A feedforward neural network computes an affine transformation followed by a non-linear activation:
$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g^{[l]}(z^{[l]})$$
where $W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$ is the weight matrix, $b^{[l]} \in \mathbb{R}^{n_l}$ is the bias vector, and $g^{[l]}$ is the activation function.

## Forward Propagation and Computational Graphs
Forward propagation evaluates intermediate node activations topologically from input layer $x = a^{[0]}$ through hidden layers to compute final prediction $\hat{y} = a^{[L]}$ and scalar loss $\mathcal{L}(\hat{y}, y)$.

## Backpropagation and the Chain Rule
Backpropagation calculates the gradient of the loss function with respect to every parameter in the network using reverse-mode automatic differentiation:
1. **Output Layer Error**:
   $$\delta^{[L]} = \nabla_{a^{[L]}} \mathcal{L} \odot g'^{[L]}(z^{[L]})$$
2. **Hidden Layer Error Propagation**:
   $$\delta^{[l]} = \left( (W^{[l+1]})^T \delta^{[l+1]} \right) \odot g'^{[l]}(z^{[l]})$$
3. **Parameter Gradients**:
   $$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^T$$
   $$\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]}$$

## Gradient Vanishing and Exploding
- **Vanishing Gradients**: When gradients propagate backward through many layers with activations whose derivatives are $< 1$ (e.g. Sigmoid $\le 0.25$, Tanh $\le 1.0$), repeated matrix multiplication causes gradients to shrink exponentially toward zero, preventing early layers from learning.
- **Exploding Gradients**: When large weights or derivatives compound across layers, gradients grow exponentially, causing numerical instability (NaN/overflow). Mitigated by **gradient clipping** ($\|g\| \leftarrow \min(1, \frac{\text{threshold}}{\|g\|}) g$) and residual connections.
