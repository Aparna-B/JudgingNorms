import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
DEFAULT_DATA_SEED = 1
np.random.seed(SEED)

def get_data_split(csv_name, cont_cat='normative', p=0, n_num=0):
    """

    csv_name: path to csv with labels, and imagename
    p: float, level of contention.
            MUST lie in between 0 and 40.

    Returns:
    train, val, test image names.

    """
    df_raw = pd.read_csv(csv_name)

    total_num = df_raw.shape[0]

    df = df_raw.copy()
    df = df.groupby('imgname').mean().reset_index()

    np.random.seed(1)
    df = df.sample(frac=1).reset_index(drop=True)
    np.random.seed(DEFAULT_DATA_SEED)
    df_high_contention = df[(df[cont_cat + '0'] >= 0.2)
                            & (df[cont_cat + '0'] <= 0.8)]

    df_low_contention = df[(df[cont_cat + '0'] < 0.2)
                           | (df[cont_cat + '0'] > 0.8)]

    # choosing test set
    test_sel_high = np.random.choice(
        df_high_contention.shape[0], int(p*600), replace=False)
    test_sel_low = np.random.choice(
        df_low_contention.shape[0], int((1-p)*600), replace=False)

    test_imgs = df_high_contention.iloc[test_sel_high].imgname.tolist(
    ) + df_low_contention.iloc[test_sel_low].imgname.tolist()

    # Dev:train + val set
    df_dev = df[~df.imgname.isin(test_imgs)]
    df_dev = df_dev.sample(frac=1).reset_index(drop=True)
    np.random.seed(DEFAULT_DATA_SEED)
    dev_imgs=df_dev.imgname.unique()

    train_imgs, val_imgs=train_test_split(dev_imgs, test_size=0.15, shuffle=True, random_state=1)
    all_imgs= [train_imgs, val_imgs, test_imgs]
    assert len(set.intersection(*map(set,all_imgs)))==0
    print(len(train_imgs) + len(val_imgs) + len(test_imgs))#,  num_images)
    return train_imgs.tolist(), val_imgs.tolist(), test_imgs

