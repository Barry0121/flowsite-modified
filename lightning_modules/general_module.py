from collections import defaultdict
from typing import Any
import torch
from torch import optim, Tensor
import lightning.pytorch as pl
from utils.logging import lg


class GeneralModule(pl.LightningModule):
    """
    Base module for PyTorch Lightning implementations that provides common functionality
    such as optimizer configuration, gradient checking, and OOM handling.
    """
    def __init__(self, args, device, model):
        """
        Initialize the GeneralModule.

        Args:
            args: Configuration parameters
            device: Computing device (CPU/GPU)
            model: The underlying neural network model
        """
        super().__init__()
        self.args = args
        self.model = model

        # Containers for logging rare/periodic values
        self.rare_logging_values = defaultdict(list)
        self.rare_logs_prepared = {}

        # Counters for tracking steps
        self.total_steps = 0
        self.total_val_steps = 0

        # Track invalid gradients during training
        self.num_invalid_gradients = 0

    def make_logs(self, logs, prefix, batch, **kwargs):
        """
        Log metrics to PyTorch Lightning's logger with appropriate configuration.

        Args:
            logs: Dictionary of metrics to log
            prefix: Prefix to add to log keys
            batch: The current batch for determining batch size
            **kwargs: Additional logging arguments
        """
        for key in logs:
            # Only show some metrics in the progress bar to avoid clutter
            prog_bar = False if 'int' in key or 'time' in key or 'all_res' in key else True
            self.log(f"{prefix}{key}", logs[key], batch_size=batch.num_graphs, prog_bar=prog_bar, **kwargs)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple of (optimizers, schedulers)
        """
        # Initialize Adam optimizer with specified learning rate
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr)

        if self.args.plateau_scheduler:
            # LR scheduler that reduces when metrics plateau
            # Note: Not fully implemented
            # raise NotImplementedError('This still needs to be integrated into the lightning training loop')
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.7,
                patience=self.args.scheduler_patience,
                min_lr=float(self.args.lr) / 100
            )

            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_kabsch_rmsd_median",  # Change this depending on the task we are optimizing for. Default to RMSD for structure.
                "interval": "epoch",
                "frequency": 1,
            },
        }
        else:
            # Create a sequence of schedulers:
            # 1. Warm-up: Linear ramp-up from lr_start*lr to lr
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.args.lr_start,
                end_factor=1.0,
                total_iters=self.args.warmup_dur
            )

            # 2. Constant: Maintain constant LR for a period
            constant = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.,
                total_iters=self.args.constant_dur
            )

            # 3. Decay: Linear decay from lr to lr_end*lr
            decay = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.,
                end_factor=self.args.lr_end,
                total_iters=self.args.decay_dur
            )

            # Combine schedulers with milestones for transitions
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, constant, decay],
                milestones=[self.args.warmup_dur, self.args.warmup_dur + self.args.constant_dur]
            )

        return [optimizer], [scheduler]

    def on_after_backward(self) -> None:
        """
        Check for invalid gradients after backward pass and zero them if needed.
        Called automatically by PyTorch Lightning after backward pass.
        """
        valid_gradients = True
        # Check all parameters for NaN or Inf gradients
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        # If invalid gradients are found, zero them out to prevent model corruption
        if not valid_gradients:
            lg(f'WARNING: NaN or Inf gradients encountered after calling backward. Setting gradients to zero.')
            self.num_invalid_gradients += 1
            self.zero_grad()

    def on_before_optimizer_step(self, optimizer):
        """
        Perform checks before optimizer step.
        Called automatically by PyTorch Lightning before optimizer step.

        Args:
            optimizer: The optimizer about to perform a step
        """
        # Check for unused parameters (parameters without gradients)
        if self.args.check_unused_params:
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    lg(f'gradients were None for {name}')

        # Check for NaN gradients
        if self.args.check_nan_grads:
            had_nan_grads = False
            for name, p in self.model.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    had_nan_grads = True
                    lg(f'gradients were nan for {name}')

            # Optionally raise exception for NaN gradients
            if had_nan_grads and self.args.except_on_nan_grads:
                raise Exception('There were nan gradients and except_on_nan_grads was set to True')

    # Wow, very smart implementation here
    def general_step_oom_wrapper(self, batch, batch_idx):
        """
        Wrapper for general_step that handles out-of-memory errors gracefully.

        Args:
            batch: Current batch of data
            batch_idx: Index of the batch

        Returns:
            Output from general_step or None if OOM error occurred
        """
        try:
            # Try to run the normal step
            return self.general_step(batch, batch_idx)
        except RuntimeError as e:
            # Handle out-of-memory errors by freeing resources
            if 'CUDA out of memory' in str(e):
                print('| WARNING: ran OOM error, skipping batch. Exception:', str(e))
                # Free gradients to reclaim memory
                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad
                # Clear CUDA cache
                torch.cuda.empty_cache()
                return None
            else:
                # Re-raise other errors
                raise e

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        """
        Override of PyTorch Lightning's backward method with OOM handling.

        Args:
            loss: The computed loss tensor
            *args: Additional arguments for backward
            **kwargs: Additional keyword arguments for backward
        """
        try:
            # Try normal backward pass
            loss.backward(*args, **kwargs)
        except RuntimeError as e:
            # Handle out-of-memory errors gracefully
            if 'CUDA out of memory' in str(e):
                print('| WARNING: ran OOM error, skipping batch. Exception:', str(e))
                # Free gradients to reclaim memory
                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad
                # Clear CUDA cache
                torch.cuda.empty_cache()
            else:
                # Re-raise other errors
                raise e