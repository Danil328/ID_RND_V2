import glob
import os
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

src_dir = '../data/'
n_folds = 4

if __name__ == '__main__':
    users = glob.glob(os.path.join(src_dir, 'train')+'/*/*') + glob.glob(os.path.join(src_dir, 'val')+'/*/*')
    df = pd.DataFrame()
    df['users'] = users

    kf = KFold(n_splits=4, shuffle=True)

    for idx, (train_index, test_index) in enumerate(kf.split(users)):
        train_users = [users[i] for i in train_index]
        val_users = [users[i] for i in test_index]
        df[f'fold_{idx}'] = df['users'].map(lambda x: 'train' if x in train_users else 'val')

    df.to_csv('cross_val_DF.csv', index=False)

    [item for user in df.users.values for item in glob.glob(user + '/*.png')]

