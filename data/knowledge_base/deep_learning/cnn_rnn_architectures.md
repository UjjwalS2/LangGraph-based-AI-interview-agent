# Convolutional and Recurrent Neural Architectures

## Convolutional Neural Networks (CNNs)
Designed for grid-structured spatial data (images, audio spectrograms):
- **Convolution Operation**: Slides a discrete learnable filter/kernel $K \in \mathbb{R}^{k_h \times k_w}$ across input feature map $X$:
  $$S(i, j) = (X * K)(i, j) = \sum_{m} \sum_{n} X(i+m, j+n) K(m, n)$$
- **Key Properties**: Parameter sharing and translation equivariance.
- **Output Dimensions**:
  $$\text{dim}_{\text{out}} = \left\lfloor \frac{\text{dim}_{\text{in}} - k + 2p}{s} \right\rfloor + 1$$
  where $p$ is padding and $s$ is stride.
- **Receptive Field**: The region in input space that influences a particular feature unit in deeper layers. Expanded via dilated convolutions or deep stacking.
- **Residual Connections (ResNet)**: Introduces skip connections $y = \mathcal{F}(x) + x$, allowing gradient signals to flow directly through identity shortcuts, resolving degradation in networks hundreds of layers deep.

## Recurrent Neural Networks (RNNs) and Gated Architectures
Designed for sequential data where output depends on prior context:
- **Vanilla RNN**:
  $$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
  Suffers severely from vanishing and exploding gradients across long time horizons $T$.
- **LSTM (Long Short-Term Memory)**:
  Maintains an explicit cell state $C_t$ regulated by three sigmoid multiplicative gates:
  1. **Forget Gate**: $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
  2. **Input Gate & Candidate**: $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$, $\tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$
  3. **Cell State Update**: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
  4. **Output Gate & Hidden State**: $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$, $h_t = o_t \odot \tanh(C_t)$
- **GRU (Gated Recurrent Unit)**:
  Simplifies LSTM by merging cell and hidden states using Reset ($r_t$) and Update ($z_t$) gates, requiring fewer parameters.
