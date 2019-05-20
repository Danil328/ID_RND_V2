import argparse
import os
import pandas as pd
import torch
import torchvision
from model.network import Model
from Dataset.id_rnd_dataset import TestAntispoofDataset
from torch.utils.data import DataLoader
from torchvision.models import resnet34, resnet101, densenet169, resnet50

PATH_MODEL = 'output/models/kek.best.h5'
BATCH_SIZE = 128
THRESHOLD = 0.1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path-images-csv', default = '../data/check_submission_data/check_images.csv', type=str, required=False)
    parser.add_argument('--path-test-dir', default = '../data/check_submission_data/check', type=str, required=False)
    parser.add_argument('--path-submission-csv', default = '../data/check_submission_data/submission.csv', type=str, required=False)
    args = parser.parse_args()

    # prepare image paths
    test_dataset_paths = pd.read_csv(args.path_images_csv)
    path_test_dir = args.path_test_dir

    paths = [
        {
            'id': row.id,
            'frame': row.frame,
            'path': os.path.join(path_test_dir, row.path)
        } for _, row in test_dataset_paths.iterrows()]

    image_dataset = TestAntispoofDataset(paths=paths)
    dataloader = DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model
    model = Model(base_model = resnet34(pretrained = False))
    model.load_state_dict(torch.load(PATH_MODEL, map_location=device))
    model = model.to(device)
    model.eval()

    # predict
    samples, frames, probabilities = [], [], []

    with torch.no_grad():
        for video, frame, batch in dataloader:
            batch = batch.to(device)
            probability = model(batch).view(-1)

            samples.extend(video)
            frames.extend(frame.numpy())
            probabilities.extend(probability.cpu().numpy())

    # save
    predictions = pd.DataFrame.from_dict({
        'id': samples,
        'frame': frames,
        'probability': probabilities})

    predictions = predictions.groupby('id').probability.mean().reset_index()
    predictions['prediction'] = predictions.probability
    predictions[['id', 'prediction']].to_csv(args.path_submission_csv, index=False)
