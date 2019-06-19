from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


def idrnd_score(target, preds, user_ids, frames):
    dataframe = pd.DataFrame({
        'user_id': user_ids,
        'frame': frames,
        'preds': preds.cpu().detach().numpy(),
        'true': target.cpu().detach().numpy()
    })

    dataframe = dataframe.groupby('user_id')['preds', 'true'].mean().reset_index()
    dataframe = dataframe[['user_id', 'preds', 'true']]
    idrnd_scores = list()
    thresholds = np.arange(0.0, 1.0, 0.01)

    for threshold in thresholds:
        temp_predicts = dataframe['preds'].values.copy()
        temp_preds = (temp_predicts > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(dataframe['true'].values, temp_preds).ravel()
        idrnd_scores.append(fp / (fp + tn) + 19 * fn / (fn + tp))

    return np.min(idrnd_scores)
