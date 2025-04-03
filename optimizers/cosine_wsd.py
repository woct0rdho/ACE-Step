from torch.optim.lr_scheduler import _LRScheduler
import torch


class CosineWSD(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, step_size, decay_length, decay_interval, eta_min=0, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.step_size = step_size
        self.decay_length = decay_length
        self.decay_interval = decay_interval
        self.eta_min = eta_min
        super(CosineWSD, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            lr = [(base_lr * self.last_epoch / self.warmup_iters) for base_lr in self.base_lrs]
        elif self.last_epoch < self.step_size:
            lr = [base_lr for base_lr in self.base_lrs]
        elif self.last_epoch <= self.step_size + self.decay_length:
            lr = [(base_lr * (0.5 ** ((self.last_epoch - self.step_size) / self.decay_interval)))
                  for base_lr in self.base_lrs]
        else:
            lr = [self.eta_min for base_lr in self.base_lrs]
        return lr


def configure_lr_scheduler(optimizer, total_steps_per_epoch, epochs=10, decay_ratio=0.9, decay_interval=1000, warmup_iters=4000):
    total_steps = total_steps_per_epoch * epochs
    step_size = total_steps * decay_ratio
    decay_length = total_steps - step_size
    decay_interval = decay_interval
    lr_scheduler = CosineWSD(
        optimizer,
        warmup_iters=warmup_iters,
        step_size=step_size,
        decay_length=decay_length,
        decay_interval=decay_interval
    )
    return [{"scheduler": lr_scheduler, "name": "CosineWSD", "interval": "step"}]
