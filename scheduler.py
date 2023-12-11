from torch.optim.lr_scheduler import LRScheduler
import math
import warnings


class CosineAnnealingLRWithWarmUp(LRScheduler):
    def __init__(
        self, optimizer, T_max, warm_up_epochs, eta_min=0, last_epoch=-1, verbose=False
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warm_up_epochs = warm_up_epochs
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch < self.warm_up_epochs:
            warmth = (self.last_epoch + 1) / self.warm_up_epochs
        else:
            warmth = 1

        if self.last_epoch == 0:
            return [warmth * group["lr"] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [
                warmth
                * (
                    self.eta_min
                    + (base_lr - self.eta_min)
                    * (1 + math.cos((self.last_epoch) * math.pi / self.T_max))
                    / 2
                )
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                warmth
                * (
                    group["lr"]
                    + (base_lr - self.eta_min)
                    * (1 - math.cos(math.pi / self.T_max))
                    / 2
                )
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            warmth
            * (
                (1 + math.cos(math.pi * self.last_epoch / self.T_max))
                / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
                * (group["lr"] - self.eta_min)
                + self.eta_min
            )
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            (min(self.last_epoch + 1, self.warm_up_epochs) / self.warm_up_epochs)
            * self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base_lr in self.base_lrs
        ]
