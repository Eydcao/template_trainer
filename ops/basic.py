import torch
from torch.nn import Sequential as Seq, Linear, GELU, LayerNorm


class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) implementation.

    This class creates a flexible MLP with customizable number of hidden layers,
    activation function, layer normalization, and residual connections.

    Args:
        input_dim (int): Dimension of the input features.
        latent_dim (int): Dimension of the hidden layers.
        output_dim (int): Dimension of the output.
        hidden_layers (int): Number of hidden layers.
        activation_mod (torch.nn.Module): Activation function module (default: GELU).
        layer_normalized (bool): Whether to apply layer normalization to the output (default: False).
        res_connection (bool): Whether to add a residual connection (default: False).
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        output_dim,
        hidden_layers,
        activation_mod=GELU(),
        layer_normalized=False,
        res_connection=False,
    ):
        super(MLP, self).__init__()
        self.res_connection = res_connection
        modules = []

        # Construct the layers of the MLP
        for l in range(hidden_layers):
            if l == 0:
                # First layer: input_dim to latent_dim
                modules.append(Linear(input_dim, latent_dim))
            else:
                # Hidden layers: latent_dim to latent_dim
                modules.append(Linear(latent_dim, latent_dim))
            # Add activation function after each linear layer
            modules.append(activation_mod)

        # Add the final output layer
        modules.append(Linear(latent_dim, output_dim))

        # Add layer normalization if specified
        if layer_normalized:
            modules.append(LayerNorm(output_dim, elementwise_affine=False))

        # Create a sequential model from the modules
        self.seq = Seq(*modules)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the MLP.
        """
        if self.res_connection:
            # If using residual connection, add input to the output
            return self.seq(x) + x
        else:
            # Otherwise, return the output directly
            return self.seq(x)
