import torch
from sklearn.metrics import confusion_matrix


def idrnd_score(y_true, y_pred):
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	return fp * (fp + tn) + 19 * fn / (fn + tp)


def idrnd_score_pytorch(target: torch.Tensor, preds: torch.Tensor) -> float:
	target = target.cpu().detach().numpy()
	preds = preds.cpu().detach().numpy().argmax(axis = 1)
	return idrnd_score(target, preds)
