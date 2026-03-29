import numpy as np

# ---------------------------------------------------------------------
# TODO 1: Compute softmax
# ---------------------------------------------------------------------
def softmax(x):
    """Compute softmax values for each row of the input array."""
    pass


class TransformerBlock:
    # ---------------------------------------------------------------------
    # TODO 2: Initialize weights
    # ---------------------------------------------------------------------
    def __init__(self, d_model=4, num_heads=2, d_ff=8, d_k=2, seed=5526):
        """
        Initialize all weights and parameters for the transformer block.

        Args:
            d_model (int): Dimension of input features, i.e., D.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feed-forward hidden layer.
            d_k (int): Dimension of Q/K/V for each head.
            seed (int): Random seed for reproducibility.
        """
        np.random.seed(seed)  # DO NOT change the seed value

        # Save parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.d_k = d_k

        # You will initialize weights here (Wq, Wk, Wv, Wo, W1, b1, W2, b2)
        # TODO: Add initialization code
        pass

    # ---------------------------------------------------------------------
    # TODO 3: Implement scaled dot-product attention
    # ---------------------------------------------------------------------
    def scaled_dot_product_attention(self, Q, K, V):
        """
        Compute scaled dot-product attention.
        
        Args:
            Q, K, V: Query, Key, and Value matrices for a single head.
        Returns:
            output: The attention output matrix.
        """
        raise NotImplementedError("Implement the scaled dot-product attention.")

    # ---------------------------------------------------------------------
    # TODO 4: Implement multi-head attention
    # ---------------------------------------------------------------------
    def multi_head_attention(self, X):
        """
        Perform multi-head self-attention over the input sequence.

        Args:
            X: Input matrix of shape (num_tokens, d_model)
        Returns:
            Output of multi-head attention (same shape as X)
        """
        raise NotImplementedError("Implement the multi-head attention mechanism.")

    # ---------------------------------------------------------------------
    # TODO 5: Implement feed-forward network
    # ---------------------------------------------------------------------
    def feed_forward(self, X):
        """
        Apply a two-layer feed-forward neural network with ReLU activation.

        Args:
            X: Input matrix (num_tokens, d_model)
        Returns:
            Output matrix (num_tokens, d_model)
        """
        raise NotImplementedError("Implement the feed-forward layer.")

    # ---------------------------------------------------------------------
    # PROVIDED: Layer Normalization (Do NOT modify)
    # ---------------------------------------------------------------------
    def layer_norm(self, x, eps=1e-6):
        """
        Normalize each token vector to have mean 0 and variance 1.
        Provided for you — do not change.
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + eps)

    # ---------------------------------------------------------------------
    # TODO 6: Combine everything into the forward pass
    # ---------------------------------------------------------------------
    def forward(self, X):
        """
        Run the full forward pass of the transformer encoder block.

        Steps:
            1. Apply multi-head attention
            2. Add & normalize (residual connection)
            3. Apply feed-forward layer
            4. Add & normalize again
        """
        raise NotImplementedError("Implement the full transformer forward pass.")
        

# ---------------------------------------------------------------------
# FIXED INPUT (all students will use this)
# ---------------------------------------------------------------------
X = np.array([
    [0.5, 1.0, 0.3, 0.7],
    [0.2, 0.9, 0.8, 0.1],
    [0.6, 0.4, 0.5, 0.9]
])

# --- Run forward pass ---
model = TransformerBlock()
output = model.forward(X)
print(np.round(output, 3))