import json
import os
import shutil

import torch
from efficientnet_pytorch import EfficientNet
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import trange, tqdm
import pretrainedmodels
from torchcontrib.optim import SWA

from Dataset.id_rnd_dataset import IDRND_dataset, make_weights_for_balanced_classes
from model.network import DoubleLossModel, DoubleLossModelTwoHead, Model
from src.tools import str2bool
from utils.loss import FocalLoss, FocalLoss2d
from utils.metrics import idrnd_score_pytorch, far_score, frr_score, bce_accuracy, idrnd_score_pytorch_for_eval

if __name__ == '__main__':
	with open('../config.json', 'r') as f:
		config = json.load(f)['train']

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	train_dataset = IDRND_dataset(mode=config['mode'], add_idrnd_v1_dataset=str2bool(config['add_idrnd_v1_dataset']),
								  use_face_detection=str2bool(config['use_face_detection']), double_loss_mode=True,
								  output_shape=config['image_resolution'])
	train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8,
							  pin_memory=True, drop_last=True)

	val_dataset = IDRND_dataset(mode=config['mode'].replace('train', 'val'), use_face_detection=str2bool(config['use_face_detection']),
								double_loss_mode=True, output_shape=config['image_resolution'])
	val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=False)

	model = DoubleLossModelTwoHead(base_model=EfficientNet.from_pretrained('efficientnet-b3')).to(device)
	# model = DoubleLossModelTwoHead(base_model=pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet')).to(device)
	# model = Model(base_model=resnet34(pretrained=True))
	summary(model, (3, config['image_resolution'], config['image_resolution']), device='cuda')

	criterion = FocalLoss(add_weight=False).to(device)
	criterion4class = CrossEntropyLoss().to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
	scheduler = ExponentialLR(optimizer, gamma=0.9)

	shutil.rmtree(config['log_path'])
	os.mkdir(config['log_path'])
	train_writer = SummaryWriter(os.path.join(config['log_path'], "train"))
	val_writer = SummaryWriter(os.path.join(config['log_path'], "val"))

	global_step = 0
	for epoch in trange(config['number_epochs']-10):
		if epoch == 5:
			train_dataset = IDRND_dataset(mode=config['mode'],
										  add_idrnd_v1_dataset=False,
										  use_face_detection=str2bool(config['use_face_detection']),
										  double_loss_mode=True,
										  output_shape=config['image_resolution'])
			train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8,
									  pin_memory=True, drop_last=True)
			criterion = FocalLoss(add_weight=True).to(device)
		model.train()
		train_bar = tqdm(train_loader)
		train_bar.set_description_str(desc=f"N epochs - {epoch}")

		scheduler.step()
		train_writer.add_scalar(tag="learning_rate", scalar_value=scheduler.get_lr()[0], global_step=epoch)
		for step, batch in enumerate(train_bar):
			global_step += 1
			image = batch['image'].to(device)
			label4class = batch['label0'].to(device)
			label = batch['label1'].to(device)

			output4class, output = model(image)
			loss4class = criterion4class(output4class, label4class)
			loss = criterion(output.squeeze(), label)
			optimizer.zero_grad()
			total_loss = loss4class*0.4 + loss*0.6
			total_loss.backward()
			optimizer.step()
			train_writer.add_scalar(tag="BinaryLoss", scalar_value=loss.item(), global_step=global_step)
			train_writer.add_scalar(tag="SoftMaxLoss", scalar_value=loss4class.item(), global_step=global_step)
			train_bar.set_postfix_str(f"Loss = {loss.item()}")
			try:
				train_writer.add_scalar(tag="idrnd_score", scalar_value=idrnd_score_pytorch(label, output), global_step=global_step)
				train_writer.add_scalar(tag="far_score", scalar_value=far_score(label, output), global_step=global_step)
				train_writer.add_scalar(tag="frr_score", scalar_value=frr_score(label, output), global_step=global_step)
				train_writer.add_scalar(tag="accuracy", scalar_value=bce_accuracy(label, output), global_step=global_step)
			except Exception:
				pass


		model.eval()
		val_bar = tqdm(val_loader)
		val_bar.set_description_str(desc=f"N epochs - {epoch}")
		outputs = []
		targets = []
		for step, batch in enumerate(val_bar):
			image = batch['image'].to(device)
			label4class = batch['label0'].to(device)
			label = batch['label1']
			with torch.no_grad():
				output4class, output = model(image)

			outputs += output.cpu().detach().view(-1).numpy().tolist()
			targets += label.cpu().detach().view(-1).numpy().tolist()

		targets = torch.tensor(targets)
		outputs = torch.tensor(outputs)
		score = idrnd_score_pytorch_for_eval(targets, outputs)
		val_writer.add_scalar(tag="idrnd_score_val", scalar_value=score, global_step=epoch)
		val_writer.add_scalar(tag="far_score_val", scalar_value=far_score(targets, outputs), global_step=epoch)
		val_writer.add_scalar(tag="frr_score_val", scalar_value=frr_score(targets, outputs), global_step=epoch)
		val_writer.add_scalar(tag="accuracy_val", scalar_value=bce_accuracy(targets, outputs), global_step=epoch)

		torch.save(model.state_dict(), f"../output/models/DoubleModelTwoHead/DoubleModel_{epoch}_{score}.pth")

	# SGD
	criterion = FocalLoss(add_weight=False).to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate']/10, weight_decay=config['weight_decay'])
	# swa = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=config['learning_rate']/20)
	scheduler = ExponentialLR(optimizer, gamma=0.9)

	for epoch in trange(config['number_epochs']-10, config['number_epochs']):
		model.train()
		train_bar = tqdm(train_loader)
		train_bar.set_description_str(desc=f"N epochs - {epoch}")

		scheduler.step()
		train_writer.add_scalar(tag="learning_rate", scalar_value=scheduler.get_lr()[0], global_step=epoch)
		for step, batch in enumerate(train_bar):
			global_step += 1
			image = batch['image'].to(device)
			label4class = batch['label0'].to(device)
			label = batch['label1'].to(device)

			output4class, output = model(image)
			loss4class = criterion4class(output4class, label4class)
			loss = criterion(output.squeeze(), label)
			optimizer.zero_grad()
			total_loss = loss4class * 0.3 + loss * 0.7
			total_loss.backward()
			optimizer.step()
			train_writer.add_scalar(tag="BinaryLoss", scalar_value=loss.item(), global_step=global_step)
			train_writer.add_scalar(tag="SoftMaxLoss", scalar_value=loss4class.item(), global_step=global_step)
			train_bar.set_postfix_str(f"Loss = {loss.item()}")
			try:
				train_writer.add_scalar(tag="idrnd_score", scalar_value=idrnd_score_pytorch(label, output),
										global_step=global_step)
				train_writer.add_scalar(tag="far_score", scalar_value=far_score(label, output),
										global_step=global_step)
				train_writer.add_scalar(tag="frr_score", scalar_value=frr_score(label, output),
										global_step=global_step)
				train_writer.add_scalar(tag="accuracy", scalar_value=bce_accuracy(label, output),
										global_step=global_step)
			except Exception:
				pass
		# swa.swap_swa_sgd()

		model.eval()
		val_bar = tqdm(val_loader)
		val_bar.set_description_str(desc=f"N epochs - {epoch}")
		outputs = []
		targets = []
		for step, batch in enumerate(val_bar):
			image = batch['image'].to(device)
			label4class = batch['label0'].to(device)
			label = batch['label1']
			with torch.no_grad():
				output4class, output = model(image)

			outputs += output.cpu().detach().view(-1).numpy().tolist()
			targets += label.cpu().detach().view(-1).numpy().tolist()

		targets = torch.tensor(targets)
		outputs = torch.tensor(outputs)
		score = idrnd_score_pytorch_for_eval(targets, outputs)
		val_writer.add_scalar(tag="idrnd_score_val", scalar_value=score, global_step=epoch)
		val_writer.add_scalar(tag="far_score_val", scalar_value=far_score(targets, outputs), global_step=epoch)
		val_writer.add_scalar(tag="frr_score_val", scalar_value=frr_score(targets, outputs), global_step=epoch)
		val_writer.add_scalar(tag="accuracy_val", scalar_value=bce_accuracy(targets, outputs), global_step=epoch)

		torch.save(model.state_dict(), f"../output/models/DoubleModelTwoHead/DoubleModel_{epoch}_{score}.pth")

