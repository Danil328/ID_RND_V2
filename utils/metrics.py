import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np


def bce_accuracy(target: torch.Tensor,
				 preds: torch.Tensor,
				 thresh: bool = 0.5) -> float:
	target = target.cpu().detach().numpy()
	preds = (preds.cpu().detach().numpy() > thresh).astype(int)
	return accuracy_score(target, preds)


def idrnd_score(y_true, y_pred):
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	return fp / (fp + tn) + 19 * fn / (fn + tp)


def idrnd_score_pytorch(target: torch.Tensor, preds: torch.Tensor, thresh=0.5) -> float:
	target = target.cpu().detach().numpy()
	preds = (preds.cpu().detach().numpy() > thresh).astype(int)
	return idrnd_score(target, preds)


def far_score(target: torch.Tensor, preds: torch.Tensor, threshold: float = 0.5) -> float:
	"""
	FAR is calculated as a fraction of negative scores exceeding your threshold.
	FAR = imposter scores exceeding threshold/all imposter scores.
	FAR = FPR = FP/(FP+TN)
	:type target: float
	"""
	target = target.cpu().detach().numpy()
	preds = (preds.cpu().detach().numpy() > threshold).astype(int)
	tn, fp, fn, tp = confusion_matrix(target, preds).ravel()
	return fp / (fp + tn)


def frr_score(target: torch.Tensor, preds: torch.Tensor, thresh: float = 0.5) -> float:
	"""
	FRR is calculated as a fraction of positive scores falling below your threshold.
	FRR = genuines scores exceeding threshold/all genuine scores genuines scores exceeding threshold = FN all genuine scores = TP+FN
	FRR = FNR = FN/(TP+FN)
	:type target: float
	"""
	target = target.cpu().detach().numpy()
	preds = (preds.cpu().detach().numpy() > thresh).astype(int)
	tn, fp, fn, tp = confusion_matrix(target, preds).ravel()
	return fn / (fn + tp)


def search_threshold(labels, predict):
	scores = list()
	thresholds = np.arange(0.05, 0.95, 0.01)
	for threshold in thresholds:
		pred = (np.array(predict.copy()) > threshold).astype(int)
		scores.append(idrnd_score(labels, pred))
	return np.min(scores), thresholds[np.argmin(scores)]
