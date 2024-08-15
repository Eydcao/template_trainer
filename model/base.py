import torch
from template_trainer.utils import Normalizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Base(torch.nn.Module):
    """
    Base class for neural network models with input and target normalization.

    This class provides a foundation for other models to inherit from, implementing
    common functionality such as data normalization and accumulation of statistics.
    """

    def __init__(self, cfg):
        """
        Initialize the Base neural network object.

        Sets up two normalizers for accumulating the mean and standard deviation
        of the input and target data, respectively.

        Args:
            cfg (OmegaConf): Configuration object containing model parameters.
        """
        super(Base, self).__init__()
        maxacum = 5e5
        input_dim = cfg.input_dim
        target_dim = cfg.target_dim
        self._inputNormalizer = Normalizer(input_dim, max_accumulations=maxacum, device=device, name="in_norm")
        self._targetNormalizer = Normalizer(target_dim, max_accumulations=maxacum, device=device, name="out_norm")

    # =====================================================================
    # Methods that need to be implemented in child classes
    # =====================================================================

    def _forward(self, input):
        """
        Abstract method for the forward pass of the model.

        This method should be implemented by child classes.

        Args:
            input (torch.Tensor): Normalized input tensor of shape (B, ..., C_in).

        Returns:
            torch.Tensor: Output tensor of shape (B, ..., C_out).

        Raises:
            NotImplementedError: If not implemented in the child class.
        """
        raise NotImplementedError("_forward needs to be implemented in child class.")

    # =====================================================================
    # Methods that do not need modifications in child classes
    # =====================================================================

    def accumulate(self, input, target):
        """
        Accumulate statistics for input and target data normalization.

        Args:
            input (torch.Tensor): Input tensor of shape (B, ..., C_in).
            target (torch.Tensor): Target tensor of shape (B, ..., C_out).
        """
        self._inputNormalizer(input, accumulate=True)
        self._targetNormalizer(target, accumulate=True)

    def report_stats(self):
        """
        Report the accumulated statistics of the input and target normalizers.
        """
        print("Input Normalizer:")
        self._inputNormalizer.report()
        print("Target Normalizer:")
        self._targetNormalizer.report()

    def forward(self, input):
        """
        Perform the forward pass of the model with normalization.

        This method normalizes the input, calls the model-specific _forward method,
        and then denormalizes the output.

        Args:
            input (torch.Tensor): Input tensor of shape (B, ..., C_in).

        Returns:
            torch.Tensor: Denormalized output tensor of shape (B, ..., C_out).

        Process:
            input -> normalize -> model -> denormalize -> output
        """
        normalized_input = self._inputNormalizer(input, accumulate=False)
        normalized_pred = self._forward(normalized_input)
        pred = self._targetNormalizer.inverse(normalized_pred)

        return pred
