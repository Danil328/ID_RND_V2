from sklearn.metrics import confusion_matrix


def get_confusion_matrix_values(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    idrnd_score = fp (fp + tn) + 19 * fn / (fn + tp)
    return idrnd_score