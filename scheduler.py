# from torch.optim.lr_scheduler import LRScheduler


# class AddWarmup(LRScheduler):
#     """A wrapper for scheduler with warm-up"""

#     def __init__(self, scheduler, warmup_epochs):
#         super().__init__(scheduler.optimizer, scheduler.last_epoch)
#         self.scheduler = scheduler
#         self.warmup_epochs = warmup_epochs
#         self.last_epoch = -1

#     def step(self):
#         current_epoch = self.last_epoch
#         if self.last_epoch < self.warmup_epochs:
#             current_epoch += 1
#         else:
#             self.scheduler.step()

#         self.last_epoch = current_epoch

#     def get_lr(self):
#         current_epoch = self.last_epoch
#         warmth = min(1.0, float(current_epoch) / self.warmup_epochs)
#         scheduler_lr = self.scheduler.get_lr()

#         return [warmth * lr for lr in scheduler_lr]

from torch.optim.lr_scheduler import LRScheduler

class AddWarmup(LRScheduler):
    """A wrapper for scheduler with warm-up"""

    def __init__(self, scheduler, warmup_epochs):
        super().__init__(scheduler.optimizer, -1)  # Initialize with -1
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [group['lr'] for group in scheduler.optimizer.param_groups]  # Save initial learning rates

    def step(self):
        if self.last_epoch < self.warmup_epochs:
            self.last_epoch += 1
        else:
            self.scheduler.step()

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Scale the learning rate based on the number of epochs
            return [base_lr * float(self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # After warm-up, use the scheduler's learning rate
            return self.scheduler.get_lr()
