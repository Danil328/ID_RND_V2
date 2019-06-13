import json
import os
import shutil

import torch
from model.efficientnet_pytorch import EfficientNet
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import trange, tqdm
import pretrainedmodels
from torchcontrib.optim import SWA

from Dataset.id_rnd_dataset import IDRND_dataset, make_weights_for_balanced_classes
from model.network import DoubleLossModel, DoubleLossModelTwoHead, Model
from src.trainLoop import str2bool
from utils.loss import FocalLoss, RobustFocalLoss2d
from utils.metrics import *
from src.train_metamodel import idrnd_score_for_eval

if __name__ == '__main__':
	with open('../config.json', 'r') as f:
		config = json.load(f)['train']

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	train_dataset = IDRND_dataset(mode=config['mode'],
								  use_face_detection=str2bool(config['use_face_detection']), double_loss_mode=True,
								  output_shape=config['image_resolution'])
	train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8,
							  pin_memory=True, drop_last=True)

	model = DoubleLossModelTwoHead(base_model=EfficientNet.from_pretrained('efficientnet-b3')).to(device)
	model.load_state_dict(torch.load('../output/models/DoubleModelTwoHead/DoubleModel_20_0.045220455749867515.pth', map_location=device))
	model.eval()

	train_bar = tqdm(train_loader)
	outputs = []
	targets = []
	user_ids = []
	frames = []
	for step, batch in enumerate(train_bar):
		image = batch['image'].to(device)
		label4class = batch['label0'].to(device)
		label = batch['label1']
		user_id = batch['user_id']
		frame = batch['frame']
		with torch.no_grad():
			output4class, output = model(image)

		outputs += output.cpu().detach().view(-1).numpy().tolist()
		targets += label.cpu().detach().view(-1).numpy().tolist()
		user_ids += user_id
		frames += frame

	df = pd.DataFrame()
	df['user_id'] = user_ids
	df['frame'] = frames
	df['probability'] = outputs
	df['target'] = targets
	# df = df.groupby('user_id')['probability', 'target'].mean().reset_index()
	# df = df[['user_id', 'probability', 'target']]

	df.to_csv("../data/train_predict_015.csv", index=False)

	val_dataset = IDRND_dataset(mode=config['mode'].replace('train', 'val'), use_face_detection=str2bool(config['use_face_detection']),
								double_loss_mode=True, output_shape=config['image_resolution'])
	val_loader = DataLoader(val_dataset, batch_size=96, shuffle=True, num_workers=8, drop_last=False)

	model.eval()
	val_bar = tqdm(val_loader)
	outputs = []
	targets = []
	user_ids = []
	frames = []
	for step, batch in enumerate(val_bar):
		image = batch['image'].to(device)
		label4class = batch['label0'].to(device)
		label = batch['label1']
		user_id = batch['user_id']
		frame = batch['frame']
		with torch.no_grad():
			output4class, output = model(image)

		outputs += output.cpu().detach().view(-1).numpy().tolist()
		targets += label.cpu().detach().view(-1).numpy().tolist()
		user_ids += user_id
		frames += frame

	df = pd.DataFrame()
	df['user_id'] = user_ids
	df['frame'] = frames
	df['probability'] = outputs
	df['target'] = targets


	df.to_csv("../data/val_predict_015.csv", index=False)

	df = df.groupby('user_id')['probability', 'target'].mean().reset_index()
	df = df[['user_id', 'probability', 'target']]
	score = idrnd_score_for_eval(df['target'].values, df['probability'].values)
	print(f"Val score = {score}")