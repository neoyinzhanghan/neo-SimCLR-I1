from torch.optim.lr_scheduler import LRScheduler


class AddWarmup(LRScheduler):
    """A wrapper for scheduler with warm-up"""

    def __init__(self, scheduler, warmup_epochs):
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmth = 1 / warmup_epochs
        super().__init__(scheduler.optimizer, scheduler.last_epoch)

    def step(self):
        self.scheduler.step()
        print("The scheduler has stepped")
        print(f"The current warmth is: {self.warmth}")
        print(f"Learning rate: {self.scheduler.get_lr()}")

    def get_lr(self):
        scheduler_lr = self.scheduler.get_lr()

        return [self.warmth * lr for lr in scheduler_lr]
