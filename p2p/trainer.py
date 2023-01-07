import logging

from detectron2.engine import DefaultTrainer, hooks
from detectron2.data import build_detection_train_loader,build_detection_test_loader
from detectron2.utils import comm

from p2p.dataset_mapper import DatasetMapper
from p2p.lr_scheduler import LRScheduler, build_lr_scheduler

import torch

class Trainer(DefaultTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)
        # some horrible stuff to get number of images and compute epoch
        total_number_of_images = len(self._trainer.data_loader.dataset.dataset.dataset)
        self.num_iters_in_epoch = int(total_number_of_images / cfg.SOLVER.IMS_PER_BATCH)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))

    @classmethod
    def build_optimizer(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.SOLVER.BASE_LR_BACKBONE,
            },
        ]
        res_optimizer = torch.optim.Adam(param_dicts, lr=cfg.SOLVER.BASE_LR)
        logger.info('Using optimizer of type: {}'.format(type(res_optimizer)))
        return res_optimizer

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def resume_or_load(self, resume=True):
        DefaultTrainer.resume_or_load(self, resume)
        self.epoch = self.iter // self.num_iters_in_epoch

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            LRScheduler(),
        ]

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


    def run_step(self):

        prev_epoch = self.epoch
        self.epoch = self.iter // self.num_iters_in_epoch

        if self.iter == 0 or prev_epoch < self.epoch:
            logger = logging.getLogger("detectron2.trainer")
            logger.info("Epoch {}".format(self.epoch))

        if prev_epoch < self.epoch:
            if hasattr(self.model, "epoch"):
                self.model.epoch = self.epoch
            for hook in self._hooks:
                if hasattr(hook, "after_epoch"):
                    hook.after_epoch()

        self._trainer.iter = self.iter
        self._trainer.run_step()

