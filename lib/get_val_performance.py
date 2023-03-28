import argparse
import math
import re
import glob

import pandas as pd
import numpy as np
import ast


import torch
import torch.nn as nn
from sklearn import metrics


def find_number(text, c):
    return re.findall(r'%s(\d+)' % c, text)


def find_float(text, c):
    return re.findall(r'%s([.\d]+)' % c, text)


def mse_loss(predictions, targets, epsilon=1e-9):
    """
    Computes MSE loss between targets and predictions.

    Input:
    predictions: (N, k) ndarray
    targets: (N, k) ndarray

    Returns: scalar
    """
    predictions = torch.from_numpy(predictions)
    targets = torch.from_numpy(targets)
    loss = nn.MSELoss()
    loss_val = []
    for i in range(len(predictions)):
        output = loss(predictions[i], targets[i])
        loss_val.append(np.sqrt(output.item()))
    return np.array(loss_val)


def compute_val_score(y_actual, y_prob):
    y_hat = np.array(y_prob > 0.5, dtype=np.int8)

    fpr, tpr, thresholds = metrics.roc_curve(y_actual,
                                             y_prob,
                                             pos_label=1)
    AUC = metrics.auc(fpr, tpr)
    ACC = metrics.accuracy_score(y_actual, y_hat)
    F1 = metrics.f1_score(y_actual, y_hat, average='macro')

    return {'Accuracy': [ACC], 'AUC': AUC, 'F1': F1}


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Get best hyperparams using validation performance.")
    parser.add_argument(
        "-d",
        "--dataset",
        help="name of dataset. Options: dress, meal, pet",
        action="store",
        type=str,
        default='dress',
        required=False)
    parser.add_argument(
        "-c",
        "--attribute_category",
        help="category to compare across conditions (0/1/2/3,"
        "where 0 refers to OR of labels)",
        action="store",
        type=int,
        default=0,
        required=False)
    parser.add_argument(
        "-t",
        "--threshold",
        help="threshold of annotator agreement to classify as violation",
        action="store",
        type=float,
        default=0.5,
        required=False)
    args = parser.parse_args()

    cat = args.attribute_category

    contention_dict = {}
    contention_dict['dress'] = 0.242
    contention_dict['meal'] = 0.4
    contention_dict['pet'] = 0.542

    df_results = []

    for csv_file in glob.glob('data_dir/{}/hyperparam_tuning_f1/*.csv'.format(
            args.dataset)):
        if 'contention_ref_normative' in csv_file:
            df_curr = pd.read_csv(csv_file)
            if 'prob' in df_curr.columns:
                df_curr.rename(columns={'prob': 'prob0'}, inplace=True)
            if 'all' in csv_file:
                df_curr['imgname'] = df_curr.apply(
                    lambda row: ast.literal_eval(row['img'])[0], axis=1)
            label_col = 'normative{}'.format(cat)

            df_labels=pd.read_csv('data_dir/{}/normative_labels.csv'.format(
            args.dataset))
            if 'descriptive' in csv_file:
                label_col = 'descriptive{}'.format(cat)
                df_labels=pd.read_csv('data_dir/{}/descriptive_labels.csv'.format(
            args.dataset))
            df_curr = df_curr.groupby('imgname').mean().reset_index()
            df_curr = df_curr.merge(df_labels, on='imgname')

            # NB: Note here we avoid pre-selecting a
            # df_curr['label'] = np.array(
            #     (df_curr[label_col] > args.threshold).values,
            #     dtype=np.int8)
            results_dict = compute_val_score(y_actual=np.array(df_curr[label_col].values>0.5,dtype=np.int8),
                                             y_prob=df_curr[
                                             'prob{}'.format(cat)])
            results_df = pd.DataFrame(results_dict)
            results_df['lr'] = find_float(csv_file, 'lr_')[0]
            results_df['batch'] = find_number(csv_file, 'batch_')[0]
            results_df['weight_decay'] = find_float(csv_file, 'weightdecay_')[0]
            results_df['seed'] = int(find_float(
                csv_file.replace('+', '_'), 'seed_')[0])
            results_df['val_label'] = label_col
            df_results.append(results_df)
    df_results = pd.concat(df_results)
    print(df_results.columns)

    all_cols_not_seed = ['lr', 'batch', 'weight_decay', 'val_label']
    df_results = df_results.groupby(
        all_cols_not_seed).mean().reset_index()

    unique_labels = df_results['val_label'].unique()
    df_best_result = []

    for unique_label in unique_labels:
        df_result_label = df_results[df_results.val_label == unique_label]
        df_best = df_result_label[
            df_result_label.F1 == df_result_label.F1.max()]
        df_best = df_best.sort_values('Accuracy')
        df_best_result.append(df_best.iloc[-1:])

    df_best_result = pd.concat(df_best_result)
    df_best_result.reset_index().to_csv(
        'data_dir/{}/best_hyperparams_tuned_5seeds.csv'.format(
            args.dataset))
    df_results.to_csv(
        'data_dir/{}/all_validation_tuned_5seeds.csv'.format(args.dataset),
        index=False)