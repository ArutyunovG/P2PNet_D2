from detectron2.config import CfgNode
from detectron2.engine import hooks

import torch

def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:

    name = cfg.SOLVER.LR_SCHEDULER_NAME
    scheduler_type = getattr(torch.optim.lr_scheduler, name)
    lr_scheduler = scheduler_type(optimizer=optimizer, **cfg.SOLVER.SCHEDULER)
    return lr_scheduler


class LRScheduler(hooks.LRScheduler):

    def after_step(self):
        self.scheduler.get_last_lr()
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)

    def after_epoch(self):
        hooks.LRScheduler.after_step(self)


