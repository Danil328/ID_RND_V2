import os
import sys
import argparse
from glob import glob

sys.path.append('..')

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.augmentations import get_test_augmentations
from src.datasets.datasets import TestDataset
from src.model.efficientnet import TwoHeadModel
from src.model.efficientnet import EfficientNetMod
from src.utils import rank_average

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_PATH = 'docker_inference/'
SIZE = 600
WORKERS = 8
BATCH_SIZE = 24
TTA = False
EF_TYPE = 'efficientnet-b3'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path-images-csv', default='../data/check_submission_data/check_images.csv', type=str, required=False)
    parser.add_argument('--path-test-dir', default='../data/check_submission_data/check', type=str, required=False)
    parser.add_argument('--path-submission-csv', default='../data/check_submission_data/submission.csv', type=str, required=False)
    args = parser.parse_args()

    image_paths = pd.read_csv(args.path_images_csv)
    test_dir = args.path_test_dir

    paths = [{'id': row.id, 'frame': row.frame, 'path': os.path.join(test_dir, row.path)} for _, row in image_paths.iterrows()]

    dataloader = DataLoader(
        TestDataset(paths, transform=get_test_augmentations(SIZE, SIZE)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS)

    model_paths = glob(BASE_PATH + '*.pth')

    for model_index, model_path in enumerate(model_paths):
        model = TwoHeadModel(EfficientNetMod.from_name(EF_TYPE)).to(DEVICE)

        state_dict = torch.load(model_path, map_location=DEVICE)
        state_dict = {k.split('.', 1)[1]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        model.eval()

        with torch.no_grad():
            if TTA:
                samples, frames, prob1, prob2 = list(), list(), list(), list()
                for id, frame, batch in dataloader:
                    batch = batch.to(DEVICE)
                    samples.extend(id)
                    frames.extend(frame.numpy())
                    prob1.extend(model(batch)[1].view(-1).cpu().numpy())
                    prob2.extend(model(batch.flip(2))[1].view(-1).cpu().numpy())

                pd.DataFrame.from_dict({
                    'id': [x + ':' + str(y) for x, y in zip(samples, frames)],
                    'probability': prob1}).to_csv(f'{BASE_PATH}temp_submission/predict1_{model_index}.csv', index=False)
                pd.DataFrame.from_dict({
                    'id': [x + ':' + str(y) for x, y in zip(samples, frames)],
                    'probability': prob2}).to_csv(f'{BASE_PATH}temp_submission/predict2_{model_index}.csv', index=False)

            else:
                samples, frames, prob = list(), list(), list()
                for id, frame, batch in dataloader:
                    batch = batch.to(DEVICE)
                    samples.extend(id)
                    frames.extend(frame.numpy())
                    prob.extend(model(batch)[1].view(-1).cpu().numpy())

                pd.DataFrame.from_dict({
                    'id': [x + ':' + str(y) for x, y in zip(samples, frames)],
                    'probability': prob}).to_csv(f'{BASE_PATH}temp_submission/predict_{model_index}.csv', index=False)

    tta_predictions_path = f'{BASE_PATH}temp_submission/submission_tta.csv'
    rank_average(f"{BASE_PATH}temp_submission/predict*.csv", tta_predictions_path)

    preds = pd.read_csv(tta_predictions_path)
    preds['frame'] = preds['id'].map(lambda x: x.split(":")[-1])
    preds['id'] = preds['id'].map(lambda x: x.split(":")[0])
    preds = preds.groupby('id').probability.mean().reset_index()
    preds['prediction'] = preds.probability
    preds[['id', 'prediction']].to_csv(args.path_submission_csv, index=False)
