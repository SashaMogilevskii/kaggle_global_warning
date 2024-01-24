import cv2
import warnings

from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import torch
import random
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2

warnings.simplefilter(action='ignore', category=FutureWarning)

def set_seed(seed=1771):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class AirDataset(Dataset):
    def __init__(self, data, is_train, size):
        self.data = data
        self.size = size
        if is_train:
            self.aug = A.Compose([
                                  A.HorizontalFlip(p=0.5),
                                  A.VerticalFlip(p=0.5),
                                  A.RandomRotate90(),
                                  A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                                                   border_mode=cv2.BORDER_REFLECT),
                                  A.OneOf([
                                      A.ElasticTransform(p=.3),
                                      A.GaussNoise(p=.3),
                                      A.GaussianBlur(),
                                      A.MotionBlur(),
                                      A.OpticalDistortion(p=0.3),
                                      A.GridDistortion(p=.1),
                                      A.IAAPiecewiseAffine(p=0.3),
                                  ], p=0.4),
                                  A.OneOf([
                                      A.HueSaturationValue(15, 25, 0),
                                      A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                                  ], p=0.3),
                                  A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                                  A.CoarseDropout(max_holes=1, max_width=int(self.size * 0.3),
                                                  max_height=int(self.size * 0.3),
                                                  mask_fill_value=0, p=0.5),
                                  A.Resize(self.size, self.size),
                                  A.Normalize(),
                                  ToTensorV2(transpose_mask=True)])


        else:
            self.aug = A.Compose([
                A.Resize(self.size, self.size),
                A.Normalize(),
                ToTensorV2(transpose_mask=True)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path_to_img = os.path.join(row['head_folder'], str(row['folder']), 'img.npy')

        image = np.load(path_to_img)
        image = (image * 255).astype(np.uint8)

        path_to_mask = os.path.join(row['head_folder'], str(row['folder']), 'human_pixel_mask.npy')
        mask = np.load(path_to_mask)

        if self.aug:
            data = self.aug(image=image, mask=mask)
            image = data['image']
            mask = data['mask']

        return image, mask

class AirDatasetMixMask(Dataset):
    def __init__(self, data, is_train, size):
        self.data = data
        self.size = size
        self.is_train = is_train
        if is_train:
            self.aug = A.Compose([
                                  A.HorizontalFlip(p=0.5),
                                  A.VerticalFlip(p=0.5),
                                  A.RandomRotate90(),
                                  A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                                                   border_mode=cv2.BORDER_REFLECT),
                                  A.OneOf([
                                      A.ElasticTransform(p=.3),
                                      A.GaussNoise(p=.3),
                                      A.GaussianBlur(),
                                      A.MotionBlur(),
                                      A.OpticalDistortion(p=0.3),
                                      A.GridDistortion(p=.1),
                                      A.IAAPiecewiseAffine(p=0.3),
                                  ], p=0.4),
                                  A.OneOf([
                                      A.HueSaturationValue(15, 25, 0),
                                      A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                                  ], p=0.3),
                                  A.Resize(self.size, self.size),
                                  A.Normalize(),
                                  ToTensorV2(transpose_mask=True)])


        else:
            self.aug = A.Compose([
                A.Resize(self.size, self.size),
                A.Normalize(),
                ToTensorV2(transpose_mask=True)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path_to_img = os.path.join(row['head_folder'], str(row['folder']), 'img.npy')
        image = np.load(path_to_img)
        image = (image * 255).astype(np.uint8)

        if self.is_train:
            path_to_mask = os.path.join(row['head_folder'], str(row['folder']), 'human_individual_mask.npy')
            mask = np.load(path_to_mask)
            averaged_array = np.mean(mask, axis=3)
            averaged_array[averaged_array >= 0.5] = 1
            mask = averaged_array

        else:
            path_to_mask = os.path.join(row['head_folder'], str(row['folder']), 'human_pixel_mask.npy')
            mask = np.load(path_to_mask)





        if self.aug:
            data = self.aug(image=image, mask=mask)
            image = data['image']
            mask = data['mask']

        return image, mask

def dice_coef(y_true, y_pred, thr, epsilon=0.001):
    y_true = y_true.flatten()

    y_pred = (y_pred > thr).astype(np.float32).flatten()

    inter = (y_true * y_pred).sum()
    den = y_true.sum() + y_pred.sum()
    dice = ((2 * inter + epsilon) / (den + epsilon))
    return dice


if __name__ == '__main__':
    # В пафы нужно .. добавить
    data = pd.read_csv('../data_6kfold.csv')
    data = data[data.head_folder.str.contains('train')]
    data = data.reset_index()

    dataset = AirDatasetMixMask(data=data,
                         is_train=True,
                         size=256)


    for i in range(100):
        dataset[i]
