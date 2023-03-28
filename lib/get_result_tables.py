import argparse
import math
import ast

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, mean_squared_error)


import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


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


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1
    FPR = FP / (FP + TN)

    return(TP, FP, TN, FN, FPR)


def get_contention_level_metrics(df, contention_levels,
                                 label_col, pred_col, prob_col,
                                 true_prob_col):
    val_dict = {}

    for contention in contention_levels:
        val_dict[contention] = []
        df_curr = df[df.contention == contention]
        TP, FP, TN, FN, _ = perf_measure(df_curr[label_col].values,
                                         df_curr[pred_col].values)
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        ACC = (TP + TN) / (TP + FP + FN + TN)
        from sklearn import metrics
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true=df_curr[label_col].values,
            y_score=df_curr[
                prob_col].values,
            pos_label=1)
        loss = metrics.auc(fpr, tpr)
        loss = f1_score(df_curr[label_col].values,
                                         df_curr[pred_col].values, 
                                         average='macro')
        # loss = metrics.average_precision_score(df_curr[label_col].values,
        #                                  df_curr[prob_col].values)

        val_dict[contention] = [ACC, loss, FPR, FNR]

    df_curr = df.copy()
    TP, FP, TN, FN, _ = perf_measure(df_curr[label_col].values,
                                     df_curr[pred_col].values)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    ACC = (TP + TN) / (TP + FP + FN + TN)
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(df_curr[label_col].values,
                                             df_curr[prob_col].values,
                                             pos_label=1)
    loss = metrics.auc(fpr, tpr)
    loss = f1_score(df_curr[label_col].values,
                                         df_curr[pred_col].values, average='macro')
    # loss = metrics.average_precision_score(df_curr[label_col].values,
    #                                      df_curr[prob_col].values)


    val_dict['All'] = [ACC, loss, FPR, FNR]

    val_df = pd.DataFrame(val_dict).T.reset_index()
    val_df.columns = ['contention', "Accuracy",
                      "Loss", "Superfluous", "Missing"]

    return val_df


def get_seed_results(df_norm, df_descr, category=0):
    df_norm['pred_dec'] = df_norm.apply(
        lambda row: row['prob{}'.format(category)] > 0.5, axis=1)
    df_descr['pred_dec'] = df_descr.apply(
        lambda row: row['prob{}'.format(category)] > 0.5, axis=1)
    df_norm['labels_dec'] = df_norm.apply(
        lambda row: row['normative{}'.format(category)] > 0.5, axis=1)
    df_descr['labels_dec'] = df_descr.apply(
        lambda row: row['normative{}'.format(category)] > 0.5, axis=1)

    # NB: Adding this does not change results, because files already
    # have defined contention.
    # df_norm['contention'] = 'Low'
    # df_norm.loc[(df_norm['normative0'] >= 0.2) & (
    #     df_norm['normative0'] <= 0.8), 'contention'] = 'High'

    # df_descr['contention'] = 'Low'
    # df_descr.loc[(df_descr['normative0'] >= 0.2) & (
    #     df_descr['normative0'] <= 0.8), 'contention'] = 'High'

    norm_df = get_contention_level_metrics(df_norm,
                                           df_norm['contention'].unique(),
                                           "labels_dec",
                                           "pred_dec",
                                           "prob{}".format(category),
                                           "normative{}".format(category))
    descr_df = get_contention_level_metrics(df_descr,
                                            df_descr['contention'].unique(),
                                            "labels_dec",
                                            "pred_dec",
                                            "prob{}".format(category),
                                            "normative{}".format(category))

    return norm_df, descr_df


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_csv_cross_seeds(train_contention_level, seedrange,
                        modelname,
                        root_dir,
                        df_labels,
                        weight=1,
                        template_str="{}_contention_{}_seed_{}_cross_{}_weight_{}.csv",
                        category=0,
                        cross=0
                        ):
    df = []
    # model_name, contention, category, seed, cross
    for seed in seedrange:
        df_curr = pd.read_csv(
            root_dir + template_str.format(
                modelname,
                train_contention_level,
                category,
                seed,
                cross
            ))
        #NB: Renaming overall prediction (norm violation)
        if 'prob' in df_curr.columns:
            df_curr.rename(columns={'prob': 'prob0'}, inplace=True)
        # formatting image name
        if 'all' in modelname:
            df_curr['imgname'] = df_curr.apply(
                lambda row: ast.literal_eval(row['img'])[0], axis=1)

        df_pred = df_curr[['imgname', 'prob0', 'prob1', 'prob2', 'prob3']]

        df_pred = df_pred.groupby('imgname').mean().reset_index()
        df_pred['seed'] = seed

        # Checking expected test set size
        assert df_pred.shape[0] == 600 or df_pred.shape[0] == 599
        df_labels = df_labels.groupby('imgname').mean().reset_index()

        df_all = df_labels.merge(df_pred, on='imgname')
        df.append(df_all)

    return pd.concat(df)


def compute_metrics_all(df_norm, df_descr, seedrange=[1, 2, 3, 4, 5],
                        train_contention=0.67, category=0):
    df_metrics = []
    for seed in seedrange:
        df_norm_curr = df_norm[df_norm.seed == seed]
        df_descr_curr = df_descr[df_descr.seed == seed]

        df_norm_curr = df_norm_curr.groupby('imgname').mean().reset_index()
        df_descr_curr = df_descr_curr.groupby('imgname').mean().reset_index()

        df_norm_curr['contention'] = 'Low'
        df_norm_curr.loc[(df_norm_curr['normative0'] >= 0.2) & (
            df_norm_curr['normative0'] <= 0.8), 'contention'] = 'High'

        df_descr_curr['contention'] = 'Low'
        df_descr_curr.loc[(df_descr_curr['normative0'] >= 0.2) & (
            df_descr_curr['normative0'] <= 0.8), 'contention'] = 'High'


        norm_df, descr_df = get_seed_results(
            df_norm_curr, df_descr_curr, category)
        norm_df['type'] = 'Train: norm, test: norm'
        descr_df['type'] = 'Train: descr, test: norm'
        df = pd.concat([norm_df, descr_df])
        df_curr_metrics = []
        for i, row in df.iterrows():
            df_curr = pd.DataFrame(row[["Loss", "type", "contention"]]).T
            df_curr.columns = ["Error Value", "type", "contention"]
            df_curr["Metric"] = "Loss"
            df_curr_metrics.append(df_curr)
            df_curr = pd.DataFrame(row[["Accuracy", "type", "contention"]]).T
            df_curr.columns = ["Error Value", "type", "contention"]
            df_curr["Metric"] = "All Error"
            df_curr_metrics.append(df_curr)
            df_curr = pd.DataFrame(
                row[["Superfluous", "type", "contention"]]).T
            df_curr.columns = ["Error Value", "type", "contention"]
            df_curr["Metric"] = "Superfluous Violations"
            df_curr_metrics.append(df_curr)
            df_curr = pd.DataFrame(row[["Missing", "type", "contention"]]).T
            df_curr.columns = ["Error Value", "type", "contention"]
            df_curr["Metric"] = "Missing Violations"
            df_curr_metrics.append(df_curr)
        df_curr_metrics = pd.concat(df_curr_metrics)
        df_curr_metrics['seed'] = seed
        df_metrics.append(df_curr_metrics)
    df_content = []
    df_metrics = pd.concat(df_metrics)
    df_metrics['Error Value'] = df_metrics['Error Value'].astype(float)
    df_metrics['Train contention'] = train_contention
    df_metrics = df_metrics.groupby(
        ['type', 'Metric', 'contention',
         'seed', 'Train contention']).mean().reset_index()
    df_content.append(df_metrics)
    # getting all results for all levels of contention
    df_content = pd.concat(df_content)
    df_content = df_content.reset_index()
    return df_content


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute significant differences between labels.")
    parser.add_argument(
        "-d",
        "--dataset",
        help="name of dataset",
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
        "-r",
        "--root_dir",
        help="root directory",
        action="store",
        type=str,
        default='data_dir/toxicity/test_output/',
        required=False)
    parser.add_argument(
        "-m",
        "--modelnames",
        help="list of model names",
        action="store",
        type=str,
        default="resnet50_all",
        required=False)
    parser.add_argument(
        "-t",
        "--traincontlevel",
        help="level of train contention",
        action="store",
        type=float,
        default=0.67,
        required=False)

    parser.add_argument(
        "-s",
        "--seedrange",
        help="range of seeds",
        action="store",
        type=str,
        default="1,2,3,4,5",
        required=False)
    parser.add_argument(
        "-t1",
        "--descr_template",
        help="template of normative csv outputs",
        action="store",
        type=str,
        # model_name, contention, category, seed,cross
        default='model_{}+batch_128_batch+{}_contention+descriptive_cat+{}_seed+{}_cross+{}_size+1_noise+0.0.csv',
        required=False)
    parser.add_argument(
        "-t2",
        "--norm_template",
        help="template of normative csv outputs",
        action="store",
        type=str,
        # model_name, contention, category, seed,cross
        default='model_{}+batch_32_batch+{}_contention+normative_cat+{}_seed+{}_cross+{}_size+1_noise+0.0.csv',
        required=False)

    args = parser.parse_args()
    df_labels = pd.read_csv(
        'data_dir/{}/normative_labels.csv'.format(args.dataset))

    # this is a common column name in all data
    df_cont = df_labels.groupby('imgname').mean().reset_index()
    df_cont['contention'] = 'Low'
    df_cont.loc[(df_cont.normative0 >= 0.2) & (
        df_cont.normative0 <= 0.8), 'contention'] = 'High'

    args.seedrange = args.seedrange.split(',')

    cat = args.attribute_category
    for modelname in args.modelnames.split(','):
        df_descr = get_csv_cross_seeds(
            train_contention_level=args.traincontlevel,
            seedrange=args.seedrange,
            modelname=modelname,
            root_dir=args.root_dir,
            template_str=args.descr_template,
            category=args.attribute_category,
            df_labels=df_labels)
        df_norm = get_csv_cross_seeds(
            train_contention_level=args.traincontlevel,
            seedrange=args.seedrange,
            modelname=modelname,
            root_dir=args.root_dir,
            template_str=args.norm_template,
            category=args.attribute_category,
            df_labels=df_labels
        )
        df_content = compute_metrics_all(
            df_norm, df_descr, args.seedrange,
            args.traincontlevel,
            category=args.attribute_category)

        df_content.to_csv(
            'cross_dataset_results/{}_auc_table.csv'.format(
                args.dataset
            ), index=False)
