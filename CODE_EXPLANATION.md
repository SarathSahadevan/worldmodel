# TinyJEPA Code Explanation

This document provides a line-by-line technical and conceptual breakdown of the `tiny_jepa.py` implementation.

---

### 1. Imports
```python
import torch
import torch.nn as nn
```
*   **Technical**: Imports the main PyTorch library and the neural network module (`nn`), which contains pre-built classes for neural network layers.
*   **Conceptual**: Loads the necessary mathematical and structural tools required to build deep learning models.

---

### 2. Class Definition
```python
class TinyJEPA(nn.Module):
```
*   **Technical**: Defines a new class `TinyJEPA` that inherits from `torch.nn.Module`.
*   **Conceptual**: This establishes our model as a standard PyTorch neural network. It inherits built-in capabilities like tracking parameters, saving/loading, and moving calculations to GPUs.

```python
    def __init__(self, input_dim=10, latent_dim=16, future_dim=5):
        super().__init__()
```
*   **Technical**: The constructor `__init__` initializes the network with customizable dimensions. `super().__init__()` properly initializes the parent `nn.Module`.
*   **Conceptual**: Sets up the blueprint for the network, defining the sizes of the raw data it will accept (past/future) and the size of the abstract representation (latent space) it will learn.

---

### 3. The Encoders
```python
        # Context Encoder (Encoder A)
        self.context_enc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
```
*   **Technical**: Creates a multi-layer perceptron (MLP) using `nn.Sequential`. It maps the input data (`input_dim`) to 64 hidden units, applies a non-linear `ReLU` activation function, and then maps down to the representation size (`latent_dim`).
*   **Conceptual**: This is the **Observer**. It takes raw data from the past and compresses it into a smaller, meaningful summary. The `ReLU` allows the network to learn complex, non-linear relationships rather than just simple flat lines.

```python
        # Target Encoder (Encoder B) - No gradient updates here!
        self.target_enc = nn.Sequential(
            nn.Linear(future_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
```
*   **Technical**: Structurally identical to the Context Encoder, but maps `future_dim` to `latent_dim`.
*   **Conceptual**: This is the **Teacher**. It looks at the actual future outcomes and creates a summary of them. Its sole job is to provide a stable target for the Predictor to aim at.

---

### 4. The Predictor
```python
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
```
*   **Technical**: Another MLP that maps from `latent_dim` back to `latent_dim`.
*   **Conceptual**: This is the **Student**. It takes the Observer's summary of the past and attempts to transform it into the Teacher's summary of the future.

---

### 5. Initialization and Freezing
```python
        if input_dim == future_dim:
            self.target_enc.load_state_dict(self.context_enc.state_dict())
```
*   **Technical**: If the inputs have the same shape, this copies the exact weights and biases from `context_enc` into `target_enc`.
*   **Conceptual**: We start the Teacher and the Observer with the exact same understanding of the world so they "speak the same language" on day one.

```python
        # Freeze target encoder
        for param in self.target_enc.parameters():
            param.requires_grad = False
```
*   **Technical**: Loops through all parameters (weights/biases) in `target_enc` and turns off gradient tracking (`requires_grad = False`).
*   **Conceptual**: This prevents the Teacher from learning via normal trial-and-error. If the Teacher was allowed to learn normally, it might cheat by producing meaningless, easy-to-predict summaries just to make the Student's job easier (a failure mode known as "representation collapse").

---

### 6. The Forward Pass
```python
    def forward(self, past, future):
```
*   **Technical**: The `forward` method defines the mathematical sequence performed at every call (e.g., when you run `model(past, future)`).
*   **Conceptual**: This is the chronological sequence of events when data flows through the network.

```python
        z_c = self.context_enc(past)
```
*   **Technical**: Passes the `past` tensor through the context encoder to get the context embedding `z_c`.
*   **Conceptual**: The Observer looks at the past and produces its summary.

```python
        z_t = self.target_enc(future) # Target embedding
```
*   **Technical**: Passes the `future` tensor through the target encoder to get the target embedding `z_t`.
*   **Conceptual**: The Teacher looks at the future and produces the "correct" summary that the Student needs to predict.

```python
        z_hat = self.predictor(z_c)   # Predicted embedding
```
*   **Technical**: Passes the context embedding `z_c` through the predictor to get the predicted embedding `z_hat`.
*   **Conceptual**: The Student takes the Observer's summary and makes its guess about what the Teacher's summary will look like.

```python
        return z_hat, z_t.detach()    # We don't backprop through Target Encoder
```
*   **Technical**: Returns the prediction and the target. `.detach()` explicitly disconnects `z_t` from the computational graph, guaranteeing no gradients will flow back into the target encoder from the loss function.
*   **Conceptual**: Hands over the Student's guess and the Teacher's answer so we can score them. Detaching is an extra safety measure to ensure the Teacher's knowledge isn't corrupted.

---

### 7. Exponential Moving Average (EMA) Update
```python
    @torch.no_grad()
    def update_target_encoder(self, momentum=0.99):
```
*   **Technical**: A custom method decorated with `@torch.no_grad()` to completely disable gradient tracking during its execution, as this is an explicit parameter update, not backpropagation. `momentum` controls the update speed.
*   **Conceptual**: The mechanism for slowly updating the Teacher. Since the Teacher is frozen, it needs a way to improve. It slowly copies the Observer's newly learned knowledge.

```python
        for param_q, param_k in zip(self.context_enc.parameters(), self.target_enc.parameters()):
```
*   **Technical**: Iterates simultaneously through the parameters of the context encoder (`param_q`) and target encoder (`param_k`).
*   **Conceptual**: Pairs up every individual connection in the Observer with the corresponding connection in the Teacher.

```python
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1.0 - momentum)
```
*   **Technical**: Performs the in-place math: `target_param = (target_param * 0.99) + (context_param * 0.01)`.
*   **Conceptual**: The Teacher retains 99% of its old knowledge but takes in 1% of the Observer's new knowledge. This keeps the Teacher stable and prevents it from radically changing its mind, which gives the Student a reliable, slowly moving target to aim for.
