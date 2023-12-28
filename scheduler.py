from torch.optim.lr_scheduler import LRScheduler


class AddWarmup(LRScheduler):
    """A wrapper for scheduler with warm-up"""

    def __init__(self, scheduler, warmup_epochs):
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        super().__init__(scheduler.optimizer, scheduler.last_epoch)

    def step(self):
        if self.last_epoch < self.warmup_epochs:
            self.last_epoch += 1
        else:
            self.scheduler.step()

    def get_lr(self):
        warmth = min(1.0, float(self.last_epoch) / self.warmup_epochs)
        scheduler_lr = self.scheduler.get_lr()

        return [warmth * lr for lr in scheduler_lr]
