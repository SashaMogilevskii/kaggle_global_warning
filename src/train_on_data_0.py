from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from torchmetrics.functional import dice
import yaml
import time
import datetime
import segmentation_models_pytorch as smp

from box import Box
from loguru import logger
from torch.cuda import amp
from utils.scripts import (set_seed)

import warnings
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import torch
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
from utils.models_zoo import CustomModel, CustomModelUnetPlusPlus, TransUnet, CustomModelDeepLabV3Plus

warnings.simplefilter(action='ignore', category=FutureWarning)


class AirDataset(Dataset):
    def __init__(self, data, is_train, size):
        self.data = data
        self.size = size
        self.is_train = is_train
        if is_train:
            self.aug = A.Compose([
                A.Resize(self.size, self.size),
                ToTensorV2(transpose_mask=True)])


        else:
            self.aug = A.Compose([
                A.Resize(self.size, self.size),
                ToTensorV2(transpose_mask=True)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path_to_img = os.path.join(row['head_folder'], str(row['folder']), 'img.npy')

        image = np.load(path_to_img)

        path_to_mask = os.path.join(row['head_folder'], str(row['folder']), 'human_pixel_mask.npy')
        mask = np.load(path_to_mask)


        if self.is_train:
            mask = np.transpose(mask[0], (1, 2, 0))

            data = self.aug(image=image, mask=mask)
            image = data['image']

            mask = data['mask']
            return image, mask
        else:
            data = self.aug(image=image, mask=mask)
            image = data['image']
            standart_mask = np.transpose(mask.copy(), (2, 0, 1))
            mask = data['mask']

            return image, mask, standart_mask


def create_train_test_loader(test_fold):
    data = pd.read_csv(config.path_data_csv)
    train_data = data[data.fold != test_fold].reset_index(drop=True)
    train_data.head_folder = train_data.head_folder.str.replace('data', 'data_0')
    test_data = data[data.fold == test_fold].reset_index(drop=True)

    train_dataset = AirDataset(data=train_data,
                               is_train=True,
                               size=config.size
                               )
    train_dataset[0]
    test_dataset = AirDataset(data=test_data,
                              is_train=False,
                              size=config.size
                              )

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers,
                              pin_memory=True,
                              drop_last=True,
                              )
    valid_loader = DataLoader(test_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              num_workers=config.num_workers,
                              pin_memory=True,
                              drop_last=False)

    return train_loader, valid_loader


date_now = datetime.datetime.now().strftime("%d_%B_%Y_%H_%M")

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    config = Box(config)

str_test_folds = [str(i) for i in config.test_folds]
path_save = os.path.join("experiment", date_now + '_' + '_'.join(str_test_folds) + '_' + config.model_name)

if not os.path.exists(path_save):
    os.makedirs(path_save)

logger.add(f"{path_save}/info__{date_now}.log",
           format="<red>{time:YYYY-MM-DD HH:mm:ss}</red>| {message}")
logger.info(f"Folder with experiment - {path_save}")
file_name = __file__
logger.info(f'file for running: {file_name}')

config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(seed=config.seed)

logger.info("----------params----------")
for param in config:
    logger.info(f"{param}: {str(config[param])}")

DiceLoss = smp.losses.DiceLoss(mode='binary', eps=2e-5, smooth=1.0)
BCELoss = smp.losses.SoftBCEWithLogitsLoss()


def criterion(y_pred, y_true):
    if config.k_bce_loss != 0:
        return config.k_bce_loss * BCELoss(y_pred, y_true) + config.k_dice_loss * DiceLoss(y_pred, y_true)
    else:
        return DiceLoss(y_pred, y_true)


def train_loop(train_loader, valid_loader, epochs, model_name_for_save, scheduler, model, optimizer, scaler):
    for epoch in range(1, epochs + 1):

        start = time.time()
        k = 0
        cur_lr = f"LR : {optimizer.param_groups[0]['lr']:.2E}"
        mloss_train, mloss_val = 0.0, 0.0

        list_y_true = None
        list_y_pred = None
        logger.info(f'Train model {model_name_for_save}')
        model.train()
        train_pbar = tqdm(train_loader, desc="Training", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        for batch in train_pbar:

            X_batch = batch[0].to(config.device)
            masks_batch = batch[1].to(config.device)

            optimizer.zero_grad()
            with amp.autocast():
                pred_masks = model(X_batch)
                loss = criterion(pred_masks, masks_batch.float())

            if torch.isnan(loss):
                logger.info("NaN значение лосса. Пропуск батча.")
                scheduler.step()
                continue

            scaler.scale(loss).backward()
            # timm.utils.adaptive_clip_grad(model.parameters(), clip_factor=0.01, eps=1e-3)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            mloss_train += loss.detach().item()

            if torch.cuda.is_available():
                train_pbar.set_postfix(gpu_load=f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB",
                                       loss=f"{loss.item():.4f}", lr=cur_lr)
            else:
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            #####
            if config.debug:
                k += 1
                if k > 5: break
            ######

        # VALID
        model.eval()
        logger.info(f'Valid model')
        valid_pbar = tqdm(valid_loader, desc="Testing", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        for batch in valid_pbar:
            X_batch = batch[0].to(config.device)
            masks_batch = batch[1].to(config.device)
            stand_masks_batch = batch[2]
            with torch.no_grad():
                pred_masks = model(X_batch)
                loss = criterion(pred_masks, masks_batch.float())
                mloss_val += loss.detach().item()
                if torch.cuda.is_available():
                    valid_pbar.set_postfix(gpu_load=f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB",
                                           loss=f"{loss.item():.4f}")
                else:
                    valid_pbar.set_postfix(loss=f"{loss.item():.4f}")

            y_preds = torch.sigmoid(pred_masks)
            if config.size != 256:
                y_preds = torch.nn.functional.interpolate(y_preds, size=256, mode='bilinear').to('cpu').numpy()

            else:
                y_preds = y_preds.to('cpu').numpy()

            if list_y_pred is None:
                list_y_pred = y_preds
                list_y_true = stand_masks_batch.to('cpu').numpy()

            else:
                list_y_pred = np.vstack((list_y_pred, y_preds))
                list_y_true = np.vstack((list_y_true, stand_masks_batch))

            ####
            if config.debug:
                k += 1
                if k > 8: break
            #####

        # Calculate metrics

        avg_train_loss = mloss_train / len(train_loader)
        avg_val_loss = mloss_val / len(valid_loader)

        logger.info(f'epoch: {epoch}')
        logger.info(cur_lr)
        logger.info("loss_train: %0.4f| loss_valid: %0.4f|" % (avg_train_loss, avg_val_loss))

        for threshold in np.arange(0.2, 0.7, 0.05):
            dice_metrics = dice(torch.Tensor(list_y_pred), torch.Tensor(list_y_true).long(), threshold=threshold)
            logger.info(f"Threshold : {threshold:.2f}\tDice_coef_kaggle : {dice_metrics:.6f}")

        elapsed_time = time.time() - start
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        logger.info(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")

        if epoch % 20 == 0:
            torch.save(model.state_dict(), f'{path_save}/{model_name_for_save}_ep_{epoch}.pt')
        torch.save(model.state_dict(), f'{path_save}/{model_name_for_save}_last_epochs.pt')

        if config.debug and epoch > 10:
            break


def main():
    for test_fold in config.test_folds:
        model_name_for_save = config.model_name + '_fold_' + str(test_fold)
        logger.info(f'START FOLD {test_fold}')
        train_loader, valid_loader = create_train_test_loader(test_fold=test_fold)
        if config.model == 'CustomModel':
            model = CustomModel(backbone_name=config.model_name,
                                weight=config.encoder_weights,
                                in_channels=3)
        elif config.model == 'PVT':
            model = TransUnet(in_channels=3)

        elif config.model == 'CustomPlus':
            model = CustomModelUnetPlusPlus(backbone_name=config.model_name,
                                            weight=config.encoder_weights,
                                            in_channels=3)
        elif config.model == 'DeepLabV3Plus':
            model = CustomModelDeepLabV3Plus(backbone_name=config.model_name,
                                             weight=config.encoder_weights,
                                             in_channels=3)

        model.to(config.device)
        print(model.__doc__)
        all_steps = len(train_loader) * config.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                       num_warmup_steps=int(all_steps // 10),
                                                                       num_training_steps=all_steps)
        scaler = amp.GradScaler()
        logger.info(f"criterion - k_dice_loss{config.k_dice_loss} + k_bce_loss{config.k_bce_loss}")
        logger.info(f"optimizer - {optimizer}")
        logger.info(f"sceduler - {scheduler}")

        train_loop(train_loader=train_loader,
                   valid_loader=valid_loader,
                   epochs=config.epochs,
                   model_name_for_save=model_name_for_save,
                   scheduler=scheduler,
                   model=model,
                   optimizer=optimizer,
                   scaler=scaler)

        logger.info(f'FINISH FOLD {test_fold}')
        logger.info(f'----------------------------')


main()
