import glob
import os
import pandas as pd
from sklearn import model_selection

SOURCE_DIR = '/mnt/hdd1/qovaxx/antospoofing/'  # dl2

if __name__ == '__main__':
    print('Start')
    user_paths = glob.glob(os.path.join(SOURCE_DIR, 'train') + '/*/*')
    folds = pd.DataFrame({'users': user_paths})
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=777)

    for index, itrain, itest in enumerate(kf.split(user_paths)):
        users_for_train = [user_paths[i] for i in itrain]
        folds[f'fold_{index}'] = folds['users'].map(lambda x: 'train' if x in users_for_train else 'val')

    folds.to_csv('folds.csv', index=False)
