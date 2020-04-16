from abc import ABC
import os
import cProfile
from contextlib import redirect_stdout
import sys
from math import ceil

import comet_ml
import torch
import torch.nn as nn
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm
import pandas as pd


from .base import TrainerBase
from models.base import PytorchModelBase
from preprocess_tools.image_utils import save_array_to_nii

load_dotenv('./.env')
RESULT_DIR_BASE = os.environ.get('RESULT_DIR')


class PytorchTrainer(TrainerBase, ABC):

    def __init__(
            self,
            model: PytorchModelBase,
            dataset_size: int,
            device_id = 0,
            optimizer=None,
            scheduler=None,
            comet_experiment: comet_ml.Experiment = None,
            checkpoint_dir=None,
            profile: bool = False,
            profile_epochs: int = 1,
    ):
        self.dataset_size = dataset_size
        self.opt = optimizer
        self.scheduler = scheduler

        EXP_ID = os.environ.get('EXP_ID')
        self.result_path = os.path.join(RESULT_DIR_BASE, EXP_ID)
        self.prob_prediction_path = None
        self.hard_prediction_path = None

        self.profile = cProfile.Profile(subcalls=False) if profile else None
        self.profile_epochs = profile_epochs
        self.profile_steps = profile_epochs * dataset_size
        self.profile_export_file_path = os.path.join(self.result_path, 'profile.stat')

        self.comet_experiment = comet_experiment

        self.model = model
        self.device_id = device_id
        # if torch.cuda.device_count() > 1:
        #     print(f'GPU count: {torch.cuda.device_count()}')
        #     self.model = nn.DataParallel(self.model)
        if torch.cuda.is_available():
            self.model.to(device_id)
            torch.backends.cudnn.benchmark = True
        print(f'Total parameters: {self.count_parameters()}')
        if checkpoint_dir is not None:
            self.load(checkpoint_dir)

        self.i_step = 0

        # for early stopping
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf


    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self):
        torch.save(
            {
                'step': self.i_step,
                'state_dict': self.model.state_dict(),
                'optimizer': self.opt.state_dict(),
            },
            os.path.join(self.result_path, 'checkpoint.pth.tar')
        )
        print(f'model saved to {self.result_path}')

    def load(self, file_path):
        checkpoint = torch.load(os.path.join(file_path, 'checkpoint.pth.tar'), map_location=f'cuda:{self.device_id}')
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.opt is not None:
            self.opt.load_state_dict(checkpoint['optimizer'])
        self.i_step = checkpoint['step'] + 1
        print(f'model loaded from {file_path}')

    def _validate(self, validation_data_generator, metric, **kwargs):
        batch_data = validation_data_generator(batch_size=1)
        label = batch_data['label']
        pred = self.model.predict(batch_data, **kwargs)
        return metric(pred, label).all_metrics()

    def _earlystopping(self, val_loss, patience=7, verbose=False, delta=0):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save()
            if verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.val_loss_min = val_loss
        elif score < self.best_score + delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {patience}')
            if self.counter >= patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save()
            if verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.val_loss_min = val_loss
            self.counter = 0


    def fit_generator(
            self,
            training_data_generator,
            validation_data_generator,
            auxiliary_data_generators,
            auxiliary_data_provider_ids,
            metric,
            batch_size,
            epoch_num,
            verbose_epoch_num,
            **kwargs,
    ):
        global batch
        step_num = epoch_num * self.dataset_size
        verbose_step_num = ceil(verbose_epoch_num * self.dataset_size)

        if self.profile is not None:
            print('Profiling...')
            self.profile.enable()
        for epoch in range(epoch_num):
            loss_sum = 0

            for batch in tqdm(range(self.dataset_size//batch_size)):
                self.i_step += batch_size

                log_dict, aux_log_dicts, loss = self.model.fit_generator(
                    training_data_generator,
                    auxiliary_data_generators,
                    self.opt,
                    batch_size=batch_size,
                )
                loss_sum += loss
                # fits on one single volume, one step = one volume

                # if self.i_step % verbose_step_num == 0:
                #     print(f'epoch: {self.i_step / self.dataset_size:.2f}', log_dict)
                #     self.save()
                #     metrics = self._validate(
                #         validation_data_generator, metric, batch_size=batch_size
                #     )
                #     if self.comet_experiment is not None:
                #         self.comet_experiment.log_metrics(
                #             log_dict, prefix='training', step=self.i_step
                #         )
                #         for log, name in zip(aux_log_dicts, auxiliary_data_provider_ids):
                #             self.comet_experiment.log_metrics(
                #                 log, prefix=f'aux_{name}', step=self.i_step
                #             )
                #         self.comet_experiment.log_metrics(
                #             metrics, prefix='validation', step=self.i_step
                #         )

                if self.i_step == self.profile_steps and self.profile is not None:
                    self.profile.disable()
                    with open(self.profile_export_file_path, 'w') as f_out:
                        with redirect_stdout(f_out):
                            self.profile.print_stats(sort='cumtime')
                    print(f"Complete profiling in {self.profile_epochs} epochs.")
                    print(f'Exported profiling stats to {self.profile_export_file_path}')
                    print("Exit by profiler")
                    sys.exit(0)
            valid_losses = []
            for _ in tqdm(range(len(validation_data_generator))):
                batch_data = validation_data_generator(batch_size=1)
                loss = self.model.validation_predict(batch_data, batch_size=1)
                valid_losses.append(loss)

            valid_loss = np.average(valid_losses)
            print(f'epoch: {epoch}, lr: {self.scheduler.get_last_lr()}')
            print(f'Training loss  : {loss_sum/(batch*batch_size)}')
            print(f'Validation loss: {valid_loss}')
            self._earlystopping(valid_loss, verbose=True)
            if self.early_stop:
                print('Early stopping!!')
                break
            self.scheduler.step()
            

    def predict_on_generator(
            self,
            data_generator,
            save_base_dir,
            metric,
            save_volume: str,
            **kwargs,
    ):
        if not os.path.exists(save_base_dir):
            os.mkdir(save_base_dir)

        self.prob_prediction_path = os.path.join(save_base_dir, f'prob_predict')
        self.hard_prediction_path = os.path.join(save_base_dir, f'hard_predict')
        if 'soft' in save_volume and not os.path.exists(self.prob_prediction_path):
            os.mkdir(self.prob_prediction_path)
        if 'hard' in save_volume and not os.path.exists(self.hard_prediction_path):
            os.mkdir(self.hard_prediction_path)

        metrics_dict = {}

        print(f'predicting on {len(data_generator)} volumes...')
        for _ in tqdm(range(len(data_generator))):
            batch_data = data_generator(batch_size=1)
            label, data_id = batch_data['label'], batch_data['data_ids'][0]
            pred = self.model.predict(batch_data, **kwargs)

            metrics = metric(pred, label).all_metrics(verbose=False)
            metrics_dict[data_id] = metrics

            self._save_volume_prediction(pred, batch_data, save_volume)

        self._save_metric_predictions(metrics_dict, save_base_dir)
        print(f'prediction result saved to {save_base_dir}')
        return metrics_dict

    def _save_volume_prediction(self, pred, batch_data, save_volume):
        # to [H, W, D, C] format
        pred = pred[0].transpose([2, 3, 1, 0])
        hard_pred = np.squeeze(pred, axis=3)
        # hard_pred = np.argmax(pred, axis=-1)

        data_id = batch_data['data_ids'][0]
        affine = batch_data['affines'][0]

        if not data_id.endswith('.nii.gz'):
            data_id = f'{data_id}.nii.gz'
        if 'soft' in save_volume:
            save_array_to_nii(pred, os.path.join(self.prob_prediction_path, data_id), affine)
        if 'hard' in save_volume:
            save_array_to_nii(hard_pred, os.path.join(self.hard_prediction_path, data_id), affine)

    @staticmethod
    def _save_metric_predictions(metrics_dict, save_base_dir):
        df = pd.DataFrame(metrics_dict).transpose()
        df = df.sort_index()
        output_file_path = os.path.join(save_base_dir, 'results.csv')
        df.to_csv(output_file_path)
