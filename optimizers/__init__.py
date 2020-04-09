import torch

from .radam import RAdam
from utils import match_kwargs


class OptimizerFactory:

    custom_optimizers = {
        'RAdam': RAdam,
    }

    def __call__(
            self,
            model_parameters,
            dataset_size,
            optimizer_type='Adam',
            # scheduler_type='MultiStepLR',
            scheduler_type='LambdaLR',
            epoch_milestones=None,
            **kwargs,
    ):
        if epoch_milestones is None:
            epoch_milestones = [50, 70]
        if optimizer_type in self.custom_optimizers.keys():
            opt_constructor = self.custom_optimizers[optimizer_type]
        else:
            opt_constructor = eval(f'torch.optim.{optimizer_type}')

        scheduler_constructor = eval(f'torch.optim.lr_scheduler.{scheduler_type}')

        # step_milestones = [n_epoch * dataset_size for n_epoch in epoch_milestones]
        lambda1 = lambda epoch: (1 - epoch/50) ** 0.9
        optimizer = opt_constructor(
            model_parameters,
            **match_kwargs(opt_constructor, **kwargs),
        )
        scheduler = scheduler_constructor(
            optimizer,
            **match_kwargs(
                scheduler_constructor,
                # milestones=epoch_milestones,
                lr_lambda=lambda1,
                **kwargs
            ),
        )
        return optimizer, scheduler
