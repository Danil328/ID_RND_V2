import json
import os
import shutil

import torch
from model.efficientnet_pytorch import EfficientNet, EfficientNetGAP
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from torchcontrib.optim import SWA

from Dataset.id_rnd_dataset import IDRND_dataset_CV
from model.network import DoubleLossModelTwoHead
from utils.loss import FocalLoss
from utils.metrics import *


def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")


def train(model_name, optim='adam'):
	train_dataset = IDRND_dataset_CV(fold=fold, mode=config['mode'],
									 add_idrnd_v1_dataset=True,
									 add_NUAA=False, aug=[0.5, 0.75, 0.25],
									 double_loss_mode=True, output_shape=config['image_resolution'])
	train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8,
							  pin_memory=True, drop_last=True)

	val_dataset = IDRND_dataset_CV(fold=fold, mode=config['mode'].replace('train', 'val'),
								   double_loss_mode=True, output_shape=config['image_resolution'])
	val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, drop_last=False)

	if model_name == 'EF':
		model = DoubleLossModelTwoHead(base_model=EfficientNet.from_pretrained('efficientnet-b3')).to(device)
		model.load_state_dict(torch.load(f"../output/models_weights/pretrained/{model_name}_{8}_1.5062978111598622_0.9967353313006619.pth"))
	elif model_name == 'EFGAP':
		model = DoubleLossModelTwoHead(base_model=EfficientNetGAP.from_pretrained('efficientnet-b3')).to(device)
		model.load_state_dict(torch.load(f"../output/models_weights/pretrained/{model_name}_{8}_1.6058124488733547_1.0.pth"))

	criterion = FocalLoss(add_weight=False).to(device)
	criterion4class = CrossEntropyLoss().to(device)

	binary_weight = 0.5
	softmax_weight = 0.5

	if optim == 'adam':
		optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
	elif optim == 'sgd':
		optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
	else:
		optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=config['learning_rate'], weight_decay=config['weight_decay'], nesterov=True)

	steps_per_epoch = train_loader.__len__()
	swa = SWA(optimizer, swa_start=config['swa_start'] * steps_per_epoch,
			  swa_freq=int(config['swa_freq'] * steps_per_epoch), swa_lr=config['learning_rate'] / 10)
	# scheduler = ExponentialLR(swa, gamma=0.9)
	scheduler = StepLR(swa, step_size=5*steps_per_epoch, gamma=0.5)

	global_step = 0
	for epoch in trange(config['number_epochs']):
		if epoch == 3:
			train_dataset = IDRND_dataset_CV(fold=fold, mode=config['mode'],
											 add_idrnd_v1_dataset=True,
											 add_NUAA=False,
											 double_loss_mode=True,
											 output_shape=config['image_resolution'],
											 aug=[0.0, 0.5, 0.0])
			train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8,
									  pin_memory=True, drop_last=True)
		model.train()
		train_bar = tqdm(train_loader)
		train_bar.set_description_str(desc=f"N epochs - {epoch}")

		for step, batch in enumerate(train_bar):
			global_step += 1
			image = batch['image'].to(device)
			label4class = batch['label0'].to(device)
			label = batch['label1'].to(device)

			output4class, output = model(image)
			loss4class = criterion4class(output4class, label4class)
			loss = criterion(output.squeeze(), label)
			swa.zero_grad()
			total_loss = loss4class * softmax_weight + loss * binary_weight
			total_loss.backward()
			swa.step()
			train_writer.add_scalar(tag="learning_rate", scalar_value=scheduler.get_lr()[0], global_step=global_step)
			train_writer.add_scalar(tag="BinaryLoss", scalar_value=loss.item(), global_step=global_step)
			train_writer.add_scalar(tag="SoftMaxLoss", scalar_value=loss4class.item(), global_step=global_step)
			train_bar.set_postfix_str(f"Loss = {loss.item()}")
			try:
				train_writer.add_scalar(tag="idrnd_score", scalar_value=idrnd_score_pytorch(label, output),
										global_step=global_step)
				train_writer.add_scalar(tag="far_score", scalar_value=far_score(label, output), global_step=global_step)
				train_writer.add_scalar(tag="frr_score", scalar_value=frr_score(label, output), global_step=global_step)
				train_writer.add_scalar(tag="accuracy", scalar_value=bce_accuracy(label, output),
										global_step=global_step)
			except Exception:
				pass

		softmax_weight *= 0.92
		binary_weight *= 1.02

		if (epoch > config['swa_start'] and epoch % 2 == 0) or (epoch == config['number_epochs'] - 1):
			swa.swap_swa_sgd()
			swa.bn_update(train_loader, model, device)
			swa.swap_swa_sgd()

		scheduler.step()
		evaluate(model, val_loader, epoch, model_name)


def evaluate(model, val_loader, epoch, model_name):
	model.eval()
	val_bar = tqdm(val_loader)
	val_bar.set_description_str(desc=f"N epochs - {epoch}")
	outputs = []
	targets = []
	user_ids = []
	frames = []
	for step, batch in enumerate(val_bar):
		image = batch['image'].to(device)
		label = batch['label1']
		user_id = batch['user_id']
		frame = batch['frame']
		with torch.no_grad():
			output4class, output = model(image)

		outputs += output.cpu().detach().view(-1).numpy().tolist()
		targets += label.cpu().detach().view(-1).numpy().tolist()
		user_ids += user_id
		frames += frame

	targets = torch.tensor(targets)
	outputs = torch.tensor(outputs)
	score = idrnd_score_pytorch_for_eval(targets, outputs)
	val_writer.add_scalar(tag="idrnd_score_val", scalar_value=score, global_step=epoch)
	val_writer.add_scalar(tag="far_score_val", scalar_value=far_score(targets, outputs), global_step=epoch)
	val_writer.add_scalar(tag="frr_score_val", scalar_value=frr_score(targets, outputs), global_step=epoch)
	val_writer.add_scalar(tag="accuracy_val", scalar_value=bce_accuracy(targets, outputs), global_step=epoch)

	user_score = idrnd_score_pytorch_for_eval_for_user(targets, outputs, user_ids, frames)
	val_writer.add_scalar(tag="idrnd_score_val_user", scalar_value=user_score, global_step=epoch)
	torch.save(model.state_dict(), f"../output/models_weights/{fold}/{model_name}_{epoch}_{score:.6f}_{user_score:.6f}.pth")


if __name__ == '__main__':
	with open('../../config.json', 'r') as f:
		config = json.load(f)['train']

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model_names = ['EF', 'EF', 'EFGAP', 'EF']
	optimizer_names = ['adam', 'adam', 'adam', 'adam']

	for fold in range(0, 5):
		config_path = f'../output/logs/fold_{fold}'
		try:
			shutil.rmtree(config_path)
		except Exception:
			pass
		os.mkdir(config_path)
		train_writer = SummaryWriter(os.path.join(config_path, "train"))
		val_writer = SummaryWriter(os.path.join(config_path, "val"))

		train(model_names[fold], optimizer_names[fold])
