import argparse
import os
import pandas as pd
import torch
import torchvision
from model.network import Model
from Dataset.id_rnd_dataset import TestAntispoofDataset
from torch.utils.data import DataLoader
from model.network import DoubleLossModel, DoubleLossModelTwoHead
from model.efficientnet_pytorch import EfficientNet
from torchvision.models import resnet34, resnet101, densenet169, resnet50

PATH_MODEL = 'for_predict/DoubleModel_11_0.01405511393746688.pth'
BATCH_SIZE = 32

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--path-images-csv', default='../data/check_submission_data/check_images.csv', type=str,
						required=False)
	parser.add_argument('--path-test-dir', default='../data/check_submission_data/check', type=str, required=False)
	parser.add_argument('--path-submission-csv', default='../data/check_submission_data/submission.csv', type=str,
						required=False)
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

	image_dataset = TestAntispoofDataset(paths=paths, use_face_detection=True)
	dataloader = DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# load model
	# model = Model(base_model=resnet34(pretrained=False))
	model = DoubleLossModelTwoHead(base_model=EfficientNet.from_name('efficientnet-b3')).to(device)
	model.load_state_dict(torch.load(PATH_MODEL, map_location=device))
	# model = torch.load(PATH_MODEL)
	model = model.to(device)
	model.eval()

	# predict
	samples, frames, probabilities = [], [], []

	with torch.no_grad():
		for video, frame, batch in dataloader:
			batch = batch.to(device)
			_, probability = model(batch)

			samples.extend(video)
			frames.extend(frame.numpy())
			probabilities.extend(probability.view(-1).cpu().numpy())

	# save
	predictions = pd.DataFrame.from_dict({
		'id': samples,
		'frame': frames,
		'probability': probabilities})

	predictions = predictions.groupby('id').probability.mean().reset_index()
	predictions['prediction'] = predictions.probability
	predictions[['id', 'prediction']].to_csv(args.path_submission_csv, index=False)
