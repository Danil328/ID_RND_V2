import os
import sys
import warnings
import pandas as pd

sys.path.append('..')
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from src.model.efficientnet import EfficientNetMod
from src.model.efficientnet import TwoHeadModel
from src.datasets.datasets import TrainDataset
from src.loss import FocalLoss
from src.metric import idrnd_score
from src.datasets.augmentations import get_train_augmentations

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 24
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 50
SIZE = 600
WORKERS = 16
BIN_COEFF = 1.02
CROSS_COEFF = 0.98
EF_TYPE = 'efficientnet-b3'
LOGS_PATH = '../output/logs'


def get_train_loader(folds, fold, steps_probas=[0.5, 0.75, 0.25]):
    train_dataset = TrainDataset(
        users=folds[folds[f'fold_{fold}'] == 'train']['users'].values,
        transform=get_train_augmentations(steps_probas, SIZE, SIZE))

    return DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True,
        drop_last=True)


def get_val_loader(folds, fold):
    val_dataset = TrainDataset(
        users=folds[folds[f'fold_{fold}'] == 'val']['users'].values,
        transform=get_train_augmentations([0.0, 0.0, 0.0], SIZE, SIZE))
    return DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        drop_last=False)


def validation_loop(model, val_loader, epoch, fold):
    model.eval()
    val_bar = tqdm(val_loader)
    val_bar.set_description_str(desc=f'Epoch: {epoch}')
    outputs = list(); targets = list(); user_ids = list(); frames = list()

    for step, batch in enumerate(val_bar):
        with torch.no_grad():
            image = batch['image'].to(DEVICE)
            cross_prediction, bin_prediction = model(image)

        outputs.extend(bin_prediction.cpu().detach().view(-1).numpy().tolist())
        targets.extend(batch['bin_label'].cpu().detach().view(-1).numpy().tolist())
        user_ids.extend(batch['user_id'])
        frames.extend(batch['frame'])

    score = idrnd_score(torch.Tensor(targets), torch.Tensor(outputs), user_ids, frames)
    print(f'Validation: {score}')

    os.makedirs(f'../output/models_weights/{fold}', exist_ok=True)
    torch.save(
        model.state_dict(), f'../output/models_weights/{fold}/{epoch}_{score:.6f}.pth')


def train_loop(fold):
    folds = pd.read_csv('datasets/folds.csv')[['users', f'fold_{fold}']]
    train_loader = get_train_loader(folds, fold)
    val_loader = get_val_loader(folds, fold)

    model = TwoHeadModel(EfficientNetMod.from_pretrained(EF_TYPE)).to(DEVICE)
    model = nn.DataParallel(model)
    focal_criterion = FocalLoss().to(DEVICE)
    crossentropy_criterion = CrossEntropyLoss().to(DEVICE)

    steps_per_epoch = len(train_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    step_learning_rate = StepLR(optimizer, step_size=5 * steps_per_epoch, gamma=0.5)

    global_step = 0
    bin_w = 0.5
    cross_w = 0.5

    for epoch in trange(EPOCHS):
        if epoch == 3:
            train_loader = get_train_loader(folds, fold, steps_probas=[0.0, 0.5, 0.0])

        model.train()
        progress_bar = tqdm(train_loader)
        progress_bar.set_description_str(desc=f'Epoch: {epoch}')

        for step, batch in enumerate(progress_bar):
            global_step += 1
            image = batch['image'].to(DEVICE)
            cross_label = batch['cross_label'].to(DEVICE)
            bin_label = batch['bin_label'].to(DEVICE)

            cross_prediction, bin_prediction = model(image)
            cross_loss = crossentropy_criterion(cross_prediction, cross_label)
            bin_loss = focal_criterion(bin_prediction.squeeze(), bin_label)

            optimizer.zero_grad()
            total_loss = cross_loss * cross_w + bin_loss * bin_w
            total_loss.backward()
            optimizer.step()

            progress_bar.set_postfix_str(f'bin_loss={bin_loss.item()}')

        cross_w = cross_w * CROSS_COEFF
        bin_w = bin_w * BIN_COEFF

        step_learning_rate.step()
        validation_loop(model, val_loader, epoch, fold)


if __name__ == '__main__':
    for current_fold in range(0, 5):
        print(f'Fold: {current_fold}')
        train_loop(current_fold)
