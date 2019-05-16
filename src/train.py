import torch
from kekas import Keker, DataOwner
from kekas.metrics import accuracy, roc_auc
from torch.utils.data import DataLoader
from torchvision.models import resnet34, resnet101

from Dataset.id_rnd_dataset import IDRND_dataset
from model.network import Model
from utils.loss import FocalLoss
from utils.metrics import idrnd_score_pytorch

if __name__ == '__main__':
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	train_dataset = IDRND_dataset(mode = 'train')
	train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 6, drop_last = True)

	val_dataset = IDRND_dataset(mode = 'val')
	val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False, num_workers = 2, drop_last = True)

	model = Model(base_model = resnet34(pretrained = True))

	dataowner = DataOwner(train_loader, val_loader, None)
	criterion = torch.nn.BCELoss()
	# criterion = FocalLoss()

	keker = Keker(model = model,
				  dataowner = dataowner,
				  criterion = criterion,
				  target_key = "label",
				  metrics = {"acc": accuracy, "roc_auc": roc_auc, "idrnd_score": idrnd_score_pytorch},
				  opt = torch.optim.Adam,
				  device = device,
				  opt_params = {"weight_decay": 1e-5})

	# find LR
	# keker.kek_lr(final_lr = 0.1, logdir = "../output/lr_find_log")

	keker.kek(lr = 1e-3,
			  epochs = 10,
			  opt = torch.optim.Adam,
			  opt_params = {"weight_decay": 1e-5},
			  sched = torch.optim.lr_scheduler.StepLR,
			  sched_params = {"step_size": 1, "gamma": 0.9},
			  logdir = "../output/logs",
			  cp_saver_params = {
				  "savedir": "../output/models/",
				  "metric": "acc",
				  "n_best": 3,
				  "prefix": "kek",
				  "mode": "max"},
			  early_stop_params = {
				  "patience": 3,
				  "metric": "acc",
				  "mode": "min",
				  "min_delta": 0
			  })
	# saving
	keker.save("../output/kek")
