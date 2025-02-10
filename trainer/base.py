import torch
from template_trainer.utils import WarmupCosineDecayScheduler
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Base_Trainer:
    def __init__(self, model, cfg, tc_rng):
        """
        Initialize the Base_Trainer.

        Args:
            model: The model to be trained.
            cfg: Configuration containing optimizer, model, and dataset configurations.
            tc_rng: Random number generator for TensorCore operations.
        """
        self.model = model
        self.cfg = cfg
        self.tc_rng = tc_rng
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
            print("Model wrapped by DataParallel", flush=True)

        self.device = device
        self.model.to(device)
        print("Model moved to {}".format(device), flush=True)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.opt.peak_lr,
            weight_decay=cfg.opt.weight_decay,
        )
        self.lr_scheduler = WarmupCosineDecayScheduler(
            optimizer=self.optimizer,
            warmup=cfg.opt.warmup_steps,
            max_iters=cfg.opt.decay_steps,
        )

        print(self.model, flush=True)
        self.train_step = 0

    # =====================================================================
    # Methods that need to be implemented in child classes
    # =====================================================================

    def _model_forward(self, data):
        """
        Perform a forward pass through the model.

        Args:
            data: PyTorch Geometric Data object containing input data.

        Returns:
            torch.Tensor: preded target.
        """
        input, _ = self.get_input_target(data)
        pred = self.model(input)
        return pred

    def get_input_target(self, data):
        """
        Extract the input and target data from the data object.

        Args:
            data: PyTorch Geometric Data object containing input and target data.

        Returns:
            tuple: (input data, target data)
        """
        return data[0], data[1]

    def accumulate(self, data):
        """
        Accumulate statistics for the model's normalizers.

        Args:
            data: PyTorch Geometric Data object containing input and target data.
        """
        data = self.move_to_device(data)
        input, target = self.get_input_target(data)
        self.model.accumulate(input, target)

    def _loss_fn(self, data):
        """
        Calculate the loss function using MSE.

        Args:
            data: PyTorch Geometric Data object containing input and target data.

        Returns:
            torch.Tensor: MSE loss value.
        """
        pred = self.get_pred(data)
        _, target = self.get_input_target(data)
        return torch.mean((pred - target) ** 2)

    def get_metrics(self, data):
        """
        Calculate the relative error for each channel and output both mean and std.

        Args:
            data: PyTorch Geometric Data object containing input data.

        Returns:
            Tuple of mean and std of the error.

        Raises:
            NotImplementedError: This method should be implemented in child classes.
        """
        raise NotImplementedError("get_metrics need to be implemented in child class.")

    def print_metrics(self, data, prefix):
        """
        Print metrics.

        Args:
            data: PyTorch Geometric Data object containing input data.
            prefix: Prefix for the metrics output.

        Raises:
            NotImplementedError: This method should be implemented in child classes.
        """
        raise NotImplementedError("print_metrics need to be implemented in child class.")

    def eval_plot(self, data, prefix):
        """
        Plot evaluation figures that are helpful for analysis.

        Args:
            data: PyTorch Geometric Data object containing input data.
            prefix: Prefix for the plot output.

        Raises:
            NotImplementedError: This method should be implemented in child classes.
        """
        raise NotImplementedError("eval_plot need to be implemented in child class.")

    def run_rollout(self, data):
        """
        Run the rollout for the model.

        Args:
            data: PyTorch Geometric Data object containing input data.

        Raises:
            NotImplementedError: This method should be implemented in child classes.
        """
        raise NotImplementedError("run_rollout need to be implemented in child class.")

    def post_process_rollout(self, data, rollout_res, bi):
        """
        Post-process the rollout results.

        Args:
            rollout_res: Rollout results.

        Raises:
            NotImplementedError: This method should be implemented in child classes.
        """
        raise NotImplementedError("post_process_rollout need to be implemented in child class.")

    def summarize_rollout(self):
        """
        Summarize the rollout results.

        Raises:
            NotImplementedError: This method should be implemented in child classes.
        """
        raise NotImplementedError("summarize_rollout need to be implemented in child class.")

    # =====================================================================
    # Methods that do not need modifications in child classes
    # =====================================================================

    def get_pred(self, data):
        """
        Get the prediction.

        Args:
            data: PyTorch Geometric Data object containing input data.

        Returns:
            Predictions as torch tensor
        """
        data = self.move_to_device(data)
        predict = self._model_forward(data)
        return predict

    def move_to_device(self, data):
        """
        Move data to the specified device.

        Args:
            data: Data to move to device (list, tuple or torch.Tensor).

        Returns:
            Data moved to device.
        """
        if isinstance(data, (list, tuple)):
            return [self.move_to_device(d) for d in data]
        else:
            return data.to(self.device)

    def iter(self, data):
        """
        Train the model for one iteration.

        Args:
            data: PyTorch Geometric Data object containing input data.
        """
        data = self.move_to_device(data)
        loss = self._loss_fn(data)

        loss.backward()

        # Gradient clipping
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.opt.gnorm_clip)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        self.train_step += 1

    def save(self, save_dir):
        """
        Save the model parameters.

        Args:
            save_dir: Directory to save the model parameters.
        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model.state_dict(), f"{save_dir}/{self.train_step}_params.pth")
        print(f"Saved to {save_dir}, step {self.train_step}")

    def restore(self, save_dir, step, restore_opt_state=True):
        """
        Restore the model parameters.

        Args:
            save_dir: Directory to restore the model parameters from.
            step: Training step to restore.
            restore_opt_state: Flag to restore optimizer state (default: True).
        """
        params_path = f"{save_dir}/{step}_params.pth"
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.load_state_dict(torch.load(params_path, map_location=device))
        print(f"Restored params from {save_dir}, step {step}")

    def get_loss(self, data):
        """
        Get the loss value.

        Args:
            data: PyTorch Geometric Data object containing input data.

        Returns:
            Loss value as np.ndarray.
        """
        data = self.move_to_device(data)
        loss = self._loss_fn(data)
        return loss

    def board_loss(self, data, prefix, board):
        """
        Log the loss to wandb.

        Args:
            data: PyTorch Geometric Data object containing input data.
            prefix: Prefix for the loss output.
            board: Flag to determine if the loss should be logged to wandb.
        """
        loss = self.get_loss(data)
        print(f"train step: {self.train_step}, {prefix}_loss: {loss}")
        if board:
            wandb.log({"step": self.train_step, f"{prefix}_loss": loss})
