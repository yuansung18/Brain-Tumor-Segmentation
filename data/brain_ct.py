import os

import nibabel as nib
import numpy as np
np.random.seed = 0

from .base import DataGeneratorBase
from .data_provider_base import DataProviderBase

from dotenv import load_dotenv

load_dotenv('./.env')

BRAIN_DIR = os.environ.get('NTU_CHALLENGE_DIR')

class BrainDataProvider(DataProviderBase):

    # _data_format = {
    #     "channels": 1,
    #     "depth": 200,
    #     "height": 200,
    #     "width": 200,
    #     "class_num": 2,
    # }

    _data_format = {
        "channels": 1,
        "depth": 76,
        "height": 182,
        "width": 182,
        "class_num": 2,
    }

    def __init__(self, args):
        self.all_ids = os.listdir(os.path.join(BRAIN_DIR, 'image'))
        self.train_ids = self.all_ids[: -len(self.all_ids) // 10]
        self.test_ids = self.all_ids[-len(self.all_ids) // 10:]

    def _get_raw_data_generator(self, data_ids, **kwargs):
        return BrainDataGenerator(data_ids, self.data_format, **kwargs)

    @property
    def data_format(self) -> dict:
        return self._data_format


class BrainDataGenerator(DataGeneratorBase):

    def __init__(self, data_ids, data_format, random=True, **kwargs):
        super().__init__(data_ids, data_format, random)
        self.data_dir  = BRAIN_DIR

    def _get_data(self, data_ids):
        batch_volume = np.zeros((
            len(data_ids),
            self.data_format['channels'],
            self.data_format['depth'],
            self.data_format['height'],
            self.data_format['width'],
        ))
        batch_label = np.zeros((
            len(data_ids),
            self.data_format['depth'],
            self.data_format['height'],
            self.data_format['width'],
        ), dtype=np.uint8)

        affines = []
        # print(data_ids)
        for idx, data_id in enumerate(data_ids):
            volume, label, affine = self._preload_get_image_and_label(data_id)

            up_idx, bottom_idx, front_idx, back_idx, left_idx, right_idx = \
                    (volume.shape[-3]-self.data_format['depth'])//2, (volume.shape[-3]+self.data_format['depth'])//2, \
                    (volume.shape[-2]-self.data_format['height'])//2, (volume.shape[-2]+self.data_format['height'])//2, \
                    (volume.shape[-1]-self.data_format['width'])//2, (volume.shape[-1]+self.data_format['width'])//2

            batch_volume[idx, :, :, :, :] = \
                volume[
                    up_idx:bottom_idx,
                    front_idx:back_idx,
                    left_idx:right_idx]

            batch_label[idx, :, :, :] = \
                label[
                    up_idx:bottom_idx,
                    front_idx:back_idx,
                    left_idx:right_idx]
            affines.append(affine)

        return {
            'volume': batch_volume,
            'label': batch_label,
            'data_ids': data_ids,
            'affines': affines,
        }

    def _preload_get_image_and_label(self, data_id):
        # Dims: (N, C, D, H, W)
        # print(data_id)
        img_path = os.path.join(self.data_dir, f"image/{data_id}")
        image_obj = nib.load(img_path)
        affine = image_obj.affine
        image = image_obj.get_fdata()
        image = np.clip(image, a_min=-1000., a_max=None)
        image = np.transpose(image, (2, 0, 1))
        label_path = os.path.join(self.data_dir, f"label/{data_id}")

        if os.path.exists(label_path):
            label = nib.load(label_path).get_fdata()
            label = np.transpose(label, (2, 0, 1))
        else:
            label = None
        return image, label, affine
