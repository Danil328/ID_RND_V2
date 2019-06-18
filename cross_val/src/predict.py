import argparse
import os
import pandas as pd
import torch
from cross_val.dataset import TestAntispoofDatasetCV
from torch.utils.data import DataLoader
from model.network import DoubleLossModelTwoHead
from model.efficientnet_pytorch import EfficientNet, EfficientNetGAP
from collections import defaultdict
from glob import glob

BASE_PATH = 'for_predict/'
output_shape = 300
BATCH_SIZE = 24
USE_TTA = False


def kaggle_bag(glob_files, loc_outfile):
	with open(loc_outfile, "w") as outfile:
		all_ranks = defaultdict(list)
		for i, glob_file in enumerate(glob(glob_files)):
			file_ranks = []
			print("parsing: {}".format(glob_file))
			# sort glob_file by first column, ignoring the first line
			lines = open(glob_file).readlines()
			lines = [lines[0]] + sorted(lines[1:])
			for e, line in enumerate(lines):
				if e == 0 and i == 0:
					outfile.write(line)
				elif e > 0:
					r = line.strip().split(",")
					file_ranks.append((float(r[1]), e, r[0]))
			for rank, item in enumerate(sorted(file_ranks)):
				all_ranks[(item[1], item[2])].append(rank)
		average_ranks = []
		for k in sorted(all_ranks):
			average_ranks.append((sum(all_ranks[k]) / len(all_ranks[k]), k))
		ranked_ranks = []
		for rank, k in enumerate(sorted(average_ranks)):
			ranked_ranks.append((k[1][0], k[1][1], rank / (len(average_ranks) - 1)))
		for k in sorted(ranked_ranks):
			outfile.write("%s,%s\n" % (k[1], k[2]))
		print("wrote to {}".format(loc_outfile))


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

	image_dataset = TestAntispoofDatasetCV(paths=paths, output_shape=output_shape)
	dataloader = DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	#device = torch.device('cpu')

	model_paths = glob(BASE_PATH+'*.pth')
	for fold, model_path in enumerate(model_paths):
		if model_path.find('EFGAP') > 0:
			model = DoubleLossModelTwoHead(base_model=EfficientNetGAP.from_name('efficientnet-b3')).to(device)
		else:
			model = DoubleLossModelTwoHead(base_model=EfficientNet.from_name('efficientnet-b3')).to(device)

		model.load_state_dict(torch.load(model_path, map_location=device))
		model = model.to(device)
		model.eval()

		if USE_TTA:
			samples, frames, probabilities1, probabilities2, probabilities3 = [], [], [], [], []
			with torch.no_grad():
				for video, frame, batch in dataloader:
					batch = batch.to(device)
					_, prob1 = model(batch)
					_, prob2 = model(batch.flip(2))  # Vertical
					_, prob3 = model(batch.flip(3))  # Horizontal
					# _, prob4 = model(batch)

					samples.extend(video)
					frames.extend(frame.numpy())
					probabilities1.extend(prob1.view(-1).cpu().numpy())
					probabilities2.extend(prob2.view(-1).cpu().numpy())
					probabilities3.extend(prob3.view(-1).cpu().numpy())

			# save
			pd.DataFrame.from_dict({
				'id': [x + ':' + str(y) for x, y in zip(samples, frames)],
				'probability': probabilities1}).to_csv(f'{BASE_PATH}temp_submission/predict1_{fold}.csv', index=False)
			pd.DataFrame.from_dict({
				'id': [x + ':' + str(y) for x, y in zip(samples, frames)],
				'probability': probabilities2}).to_csv(f'{BASE_PATH}temp_submission/predict2_{fold}.csv', index=False)
			pd.DataFrame.from_dict({
				'id': [x + ':' + str(y) for x, y in zip(samples, frames)],
				'probability': probabilities3}).to_csv(f'{BASE_PATH}temp_submission/predict3_{fold}.csv', index=False)

		else:
			samples, frames, probabilities = [], [], []
			with torch.no_grad():
				for video, frame, batch in dataloader:
					batch = batch.to(device)
					_, probability = model(batch)

					samples.extend(video)
					frames.extend(frame.numpy())
					probabilities.extend(probability.view(-1).cpu().numpy())

			# save
			pd.DataFrame.from_dict({
				'id': [x + ':' + str(y) for x, y in zip(samples, frames)],
				'probability': probabilities}).to_csv(f'{BASE_PATH}temp_submission/predict_{fold}.csv', index=False)

	# Rank Average
	kaggle_bag(f"{BASE_PATH}temp_submission/predict*.csv", f'{BASE_PATH}temp_submission/submission_tta.csv')
	predictions = pd.read_csv(f'{BASE_PATH}temp_submission/submission_tta.csv')
	predictions['frame'] = predictions['id'].map(lambda x: x.split(":")[-1])
	predictions['id'] = predictions['id'].map(lambda x: x.split(":")[0])
	predictions = predictions.groupby('id').probability.mean().reset_index()
	predictions['prediction'] = predictions.probability
	predictions[['id', 'prediction']].to_csv(args.path_submission_csv, index=False)
