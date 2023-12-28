# from torch.optim.lr_scheduler import LRScheduler


# class AddWarmup(LRScheduler):
#     """A wrapper for scheduler with warm-up"""

#     def __init__(self, scheduler, warmup_epochs):
#         super().__init__(scheduler.optimizer, scheduler.last_epoch)
#         self.scheduler = scheduler
#         self.warmup_epochs = warmup_epochs

#     def step(self):
#         if self.last_epoch < self.warmup_epochs:
#             self.last_epoch += 1
#         else:
#             self.scheduler.step()

#     def get_lr(self):
#         warmth = min(1.0, float(self.last_epoch) / self.warmup_epochs)
#         scheduler_lr = self.scheduler.get_lr()

#         return [warmth * lr for lr in scheduler_lr]


from torch.optim.lr_scheduler import LRScheduler

class AddWarmup(LRScheduler):
    """A wrapper for scheduler with warm-up"""

    def __init__(self, scheduler, warmup_epochs):
        self.scheduler = scheduler
        self.optimizer = scheduler.optimizer  # Set optimizer directly
        self.warmup_epochs = warmup_epochs
        self.last_epoch = -1  # Initialize last_epoch directly

        print(f"Initialized AddWarmup with warmup_epochs: {self.warmup_epochs}")  # Debug print

    def step(self):
        current_epoch = self.last_epoch + 1
        print(f"Step called with current_epoch: {current_epoch}, warmup_epochs: {self.warmup_epochs}")  # Debug print

        if current_epoch <= self.warmup_epochs:
            self._last_lr = [base_lr * current_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            self.scheduler.step()
            self._last_lr = self.scheduler.get_last_lr()
        self.last_epoch = current_epoch

    def get_lr(self):
        if self.last_epoch <= self.warmup_epochs:
            return [base_lr * float(self.last_epoch) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            return self.scheduler.get_lr()
