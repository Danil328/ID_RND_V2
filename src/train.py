import json
import os
import shutil

import torch
from kekas import Keker, DataOwner
from kekas.metrics import roc_auc
from torch.utils.data import DataLoader
from torchvision.models import resnet34, resnet101, densenet169, resnet50
from torch.optim.lr_scheduler import ExponentialLR
from pretrainedmodels import se_resnet101
from Dataset.id_rnd_dataset import IDRND_dataset, make_weights_for_balanced_classes
from model.network import Model
from utils.loss import FocalLoss, WeightedBCELoss
from utils.metrics import idrnd_score_pytorch, far_score, frr_score, bce_accuracy, search_threshold
from utils.sheduler import CyclicCosAnnealingLR
from model.fishnet.net_factory import fishnet99
from torchsummary import summary
from src.tools import str2bool

if __name__ == '__main__':
	with open('../config.json', 'r') as f:
		config = json.load(f)['train']

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	train_dataset = IDRND_dataset(mode=config['mode'], add_idrnd_v1_dataset=str2bool(config['add_idrnd_v1_dataset']),
								  use_face_detection=str2bool(config['use_face_detection']))
	if str2bool(config['use_sampler']):
		weights = make_weights_for_balanced_classes(train_dataset)
		weights_tensor = torch.DoubleTensor(weights)
		sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
		train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, drop_last=True,
								  sampler=sampler)
	else:
		train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, drop_last=True,
								  pin_memory=True)


	val_dataset = IDRND_dataset(mode=config['mode'].replace('train', 'val'), use_face_detection=False)
	val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, drop_last=False)

	# model = Model(base_model = fishnet99())
	model = Model(base_model=resnet34(pretrained=True))
	summary(model, (3, 224, 224), device="cpu")

	dataowner = DataOwner(train_loader, val_loader, None)
	# criterion = torch.nn.BCELoss()
	# criterion = WeightedBCELoss(weights=[0.49, 0.51]) Не обучается
	criterion = FocalLoss()

	shutil.rmtree('../output/logs')
	os.mkdir('../output/logs')

	keker = Keker(model=model,
				  dataowner=dataowner,
				  criterion=criterion,
				  target_key="label",
				  metrics={"acc": bce_accuracy, "idrnd_score": idrnd_score_pytorch, "FAR": far_score,
						   # "roc_auc": roc_auc,
						   "FRR": frr_score},
				  opt=torch.optim.Adam,
				  device=device,
				  opt_params={"weight_decay": 1e-5})

	# find LR
	# keker.kek_lr(final_lr = 0.1, logdir = "../output/lr_find_log")

	keker.kek(lr=config['learning_rate'],
			  epochs=config['number_epochs'],
			  opt=torch.optim.Adam,
			  opt_params={"weight_decay": config['weight_decay']},
			  # sched=CyclicCosAnnealingLR,
			  sched=ExponentialLR,
			  # sched_params={"milestones": [12, 29], "eta_min": 1e-8},
			  sched_params={"gamma": 0.9},
			  logdir="../output/logs",
			  cp_saver_params={
				  "savedir": "../output/models/",
				  "metric": "idrnd_score",
				  "n_best": 3,
				  "prefix": "kek",
				  "mode": "min"},
			  # early_stop_params={
			  #     "patience": 5,
			  #     "metric": "idrnd_score",
			  #     "mode": "min",
			  #     "min_delta": 0}
			  )
	# saving
	keker.save("../output/kek")

	keker.load("../output/kek")

	# Search best threshold
	predict = keker.predict_loader(val_loader)
	labels = val_dataset.labels
	best_idrnd_score, threshold = search_threshold(labels, predict)
	print(f"Best score - {best_idrnd_score}\nBest threshold - {threshold}")
