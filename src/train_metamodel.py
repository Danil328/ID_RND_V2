import cv2
import numpy as np
import pandas as pd
import glob
import os
import torch
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from functools import reduce
from Dataset.id_rnd_dataset import TestAntispoofDataset
from model.efficientnet_pytorch import EfficientNet
from model.network import DoubleLossModelTwoHead
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV

PATH_MODEL = '../for_predict/DoubleModel_14_0.00701377848436672.pth'


def idrnd_score(y_true, y_pred):
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	return fp / (fp + tn) + 19 * fn / (fn + tp)


def idrnd_score_for_eval(target, preds, thresholds=np.arange(0.1, 1.0, 0.01)) -> float:
	scores = []
	for thresh in thresholds:
		temp_preds = preds.copy()
		temp_preds = (temp_preds > thresh).astype(int)
		scores.append(idrnd_score(target, temp_preds))
	return np.min(scores)


if __name__ == '__main__':
	path_to_data = '../data'
	masks = glob.glob(os.path.join(path_to_data, 'val', '2dmask/*/*.png'))
	printed = glob.glob(os.path.join(path_to_data, 'val', 'printed/*/*.png'))
	replay = glob.glob(os.path.join(path_to_data, 'val', 'replay/*/*.png'))
	real = glob.glob(os.path.join(path_to_data, 'val', 'real/*/*.png'))

	images = masks + printed + replay + real
	labels = [1] * len(masks + printed + replay) + [0] * len(real)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# load model
	model = DoubleLossModelTwoHead(base_model=EfficientNet.from_name('efficientnet-b3')).to(device)
	model.load_state_dict(torch.load(PATH_MODEL, map_location=device))
	model = model.to(device)
	model.eval()


	df = pd.DataFrame()
	df['path'] = images
	df['user_id'] = df['path'].map(lambda x: x.split('/')[-2].split('_')[-1])
	df['frame'] = df['path'].map(lambda x: x[-6:-4])
	df['label'] = labels

	predicts = []
	with torch.no_grad():
		for path in tqdm(df.path.values):
			img = cv2.imread(path)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = cv2.resize(img, (300, 300)) / 255.
			img = np.moveaxis(img, -1, 0)

			img = torch.tensor(img, dtype=torch.float).to(device)
			_, probability = model(img.unsqueeze(0))
			predicts += probability.cpu().detach().view(-1).numpy().tolist()

	df['probability'] = predicts

	frame1 = df[df['frame'] == '01'][['user_id', 'label', 'probability']]
	frame1.columns = ['user_id', 'label', 'frame1']
	frame2 = df[df['frame'] == '02'][['user_id', 'probability']]
	frame2.columns = ['user_id', 'frame2']
	frame3 = df[df['frame'] == '03'][['user_id', 'probability']]
	frame3.columns = ['user_id', 'frame3']
	frame4 = df[df['frame'] == '04'][['user_id', 'probability']]
	frame4.columns = ['user_id', 'frame4']
	frame5 = df[df['frame'] == '05'][['user_id', 'probability']]
	frame5.columns = ['user_id', 'frame5']

	dfs = [frame1, frame2, frame3, frame4, frame5]
	df_final = reduce(lambda left, right: pd.merge(left, right, on='user_id'), dfs)

	# Baseline
	baseline = df.groupby('user_id')['probability', 'label'].median().reset_index()
	print(f"Baseline score: {idrnd_score_for_eval(baseline.label.values, baseline.probability.values)}")

	lr_score = cross_val_score(estimator=LogisticRegression(class_weight={0:1, 1:1}),
					X=df_final[['frame1', 'frame2', 'frame3', 'frame4', 'frame5']].values,
					y=df_final['label'].values,
					cv=10,
					scoring=make_scorer(idrnd_score_for_eval)).mean()
	print(f"LR score = {lr_score}")

	clf = GridSearchCV(KNeighborsClassifier(), {'n_neighbors':range(5, 10)}, cv=5, scoring=make_scorer(idrnd_score_for_eval))
	clf.fit(df_final[['frame1', 'frame2', 'frame3', 'frame4', 'frame5']].values, df_final['label'].values)

	# Votes
	df_copy = df.copy()
	df_copy['probability'] = df_copy['probability'].map(lambda x: 1.0 if x > 0.5 else 0.0)
	baseline = df_copy.groupby('user_id')['probability', 'label'].mean().reset_index()
	print(f"Baseline score: {idrnd_score_for_eval(baseline.label.values, baseline.probability.values)}")
