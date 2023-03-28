import argparse
import math

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, mean_squared_error)


import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import warnings

batch_size_dict={}
batch_size_dict['albert-base-v2']={}
batch_size_dict['roberta-base']={}

#NB: this is chosen based on validation performance
batch_size_dict['albert-base-v2'][0]=32
batch_size_dict['albert-base-v2'][1]=32
batch_size_dict['roberta-base'][0]=32
batch_size_dict['roberta-base'][1]=32


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
                                 label_col, pred_col, prob_col, true_prob_col):
    val_dict = {}

    for contention in contention_levels:
        val_dict[contention] = []
        df_curr = df[df.contention == contention]
        TP, FP, TN, FN, _ = perf_measure(df_curr[label_col].values,
                                         df_curr[pred_col].values)
        # Sensitivity, hit rate, recall, or true positive rate
        try:
            TPR = TP / (TP + FN)
            # Specificity or true negative rate
            TNR = TN / (TN + FP)
            # Fall out or false positive rate
            FPR = FP / (FP + TN)
            # False negative rate
            FNR = FN / (TP + FN)
            ACC = (TP + TN) / (TP + FP + FN + TN)
        except:
            ACC = (TP + TN) / (TP + FP + FN + TN)
            TNR=TPR=FPR=FNR=np.nan

        from sklearn import metrics
        fpr, tpr, thresholds = metrics.roc_curve(df_curr[label_col].values,
                                                 df_curr[prob_col].values,
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
                                         df_curr[pred_col].values, 
                                         average='macro')
    # loss= metrics.average_precision_score(df_curr[label_col].values,
    #                                      df_curr[prob_col].values)




    val_dict['All'] = [ACC, loss, FPR, FNR]

    val_df = pd.DataFrame(val_dict).T.reset_index()
    val_df.columns = ['contention', "Accuracy",
                      "Loss", "Superfluous", "Missing"]

    return val_df


def get_seed_results(df_norm, df_descr):
    df_norm['pred_dec'] = df_norm.apply(lambda row: row['pred0'] > 0.5, axis=1)
    df_descr['pred_dec'] = df_descr.apply(
        lambda row: row['pred0'] > 0.5, axis=1)
    df_norm['labels_dec'] = df_norm.apply(
        lambda row: row['normative0'] > 0.5, axis=1)
    df_descr['labels_dec'] = df_descr.apply(
        lambda row: row['normative0'] > 0.5, axis=1)

    df_norm['contention'] = 'Low'
    df_norm.loc[(df_norm['normative0'] >= 0.2) & (
        df_norm['normative0'] <= 0.8), 'contention'] = 'High'

    df_descr['contention'] = 'Low'
    df_descr.loc[(df_descr['normative0'] >= 0.2) & (
        df_descr['normative0'] <= 0.8), 'contention'] = 'High'

    norm_df = get_contention_level_metrics(df_norm,
                                           df_descr['contention'].unique(),
                                           "labels_dec",
                                           "pred_dec",
                                           "pred0", "normative0")
    descr_df = get_contention_level_metrics(df_descr,
                                            df_descr['contention'].unique(),
                                            "labels_dec",
                                            "pred_dec",
                                            "pred0",
                                            "normative0")

    return norm_df, descr_df


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_csv_cross_seeds(train_contention_level, seedrange,
                        modelname, cross,
                        root_dir,
                        batch_size,
                        weight=1,
                       template_str="contention_ref_normative_{}_contention_{}_bs_{}_seed_{}_cross_{}_weight_{}_size_1_noise_0.csv",
                       df_labels=None):
    df = []
    for seed in seedrange:
        df_curr = pd.read_csv(
            root_dir + template_str.format(
                modelname,
                train_contention_level,
                batch_size,
                seed,
                cross,
                weight))
        for i in range(4):
            df_curr['pred{}'.format(i)] = df_curr.apply(lambda row: float(
                sigmoid(row['prediction{}'.format(i)])), axis=1)
        df_pred = df_curr[['text', 'pred0', 'pred1', 'pred2', 'pred3']]
        # we aggregate later on as well, so this step is optional
        df_pred = df_pred.groupby('text').mean().reset_index()
        df_pred['seed'] = seed
        df_all = df_labels.merge(df_pred, on='text')
        df.append(df_all)

    return pd.concat(df)


def compute_metrics_subgroup(df_norm, df_descr, seedrange=[1, 2, 3, 4, 5],
                        train_contention=0.67):
    df_metrics = []
    descr_acc={}
    norm_acc={}
    
    descr_acc['male']=[]
    norm_acc['male']=[]
    descr_acc['female']=[]
    norm_acc['female']=[]
    df_annot = pd.read_csv('civilcomments_files/train.csv')
    df_annot = df_annot.drop_duplicates(subset=['comment_text'])
    for seed in seedrange:
        df_norm_curr = df_norm[df_norm.seed == seed]
        df_descr_curr = df_descr[df_descr.seed == seed]

        df_norm_curr = df_norm_curr.groupby('text').mean().reset_index()
        df_descr_curr = df_descr_curr.groupby('text').mean().reset_index()

        df_norm_curr = df_norm_curr.merge(df_annot[['comment_text','male','female']], 
                                          left_on='text', right_on='comment_text',how='inner')
        df_descr_curr = df_descr_curr.merge(df_annot[['comment_text','male','female']], 
                                          left_on='text', right_on='comment_text', how='inner')
        
        for sex in ['male','female']:
            # NB: This dataset only has binary sex annotations
            not_gender = list(set(['male','female'])-set([sex]))

            # Considering a text to mention a specific sex mention if over
            # 50% of the annotators labelled it as such.
            df_norm_=df_norm_curr[(df_norm_curr[sex]>0.5)]
            df_descr_=df_descr_curr[(df_descr_curr[sex]>0.5)]
            print(df_norm_.shape,df_descr_.shape,sex,df_norm_curr.shape)
            norm_df, descr_df = get_seed_results(df_norm_, df_descr_)
            # This is to get violation prediction rates
            df_norm_['pred_dec'] = df_norm_.apply(lambda row: row['pred0'] > 0.5, axis=1)
            df_descr_['pred_dec'] = df_descr_.apply(
                 lambda row: row['pred0'] > 0.5, axis=1)
            # descr_acc[sex].append(df_descr_['pred_dec'].mean())
            # norm_acc[sex].append(df_norm_['pred_dec'].mean())
            descr_acc[sex].append(descr_df[descr_df.contention=='All']['Accuracy'].mean())
            norm_acc[sex].append(norm_df[norm_df.contention=='All']['Accuracy'].mean())

    for sex in ['male','female']:
        print('Sex: {}, Normative acc: {}, Descriptive acc: {}'.format(
              sex,
              np.mean(norm_acc[sex]),
              np.mean(descr_acc[sex])))
        

def compute_metrics_all(df_norm, df_descr, seedrange=[1, 2, 3, 4, 5],
                        train_contention=0.67):
    df_metrics = []
    for seed in seedrange:
        df_norm_curr = df_norm[df_norm.seed == seed]
        df_descr_curr = df_descr[df_descr.seed == seed]

        df_norm_curr = df_norm_curr.groupby('text').mean().reset_index()
        df_descr_curr = df_descr_curr.groupby('text').mean().reset_index()

        df_norm_curr['contention'] = 'Low'
        df_norm_curr.loc[(df_norm_curr['normative0'] >= 0.2) & (
            df_norm_curr['normative0'] <= 0.8), 'contention'] = 'High'

        df_descr_curr['contention'] = 'Low'
        df_descr_curr.loc[(df_descr_curr['normative0'] >= 0.2) & (
            df_descr_curr['normative0'] <= 0.8), 'contention'] = 'High'

        norm_df, descr_df = get_seed_results(df_norm_curr, df_descr_curr)
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
        help="path to csv with descriptive, normative, and context labels",
        action="store",
        type=str,
        default='data_dir/toxicity/normative_labels.csv',
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
        default='data_dir/toxicity/output_main/test_output/',
        required=False)
    parser.add_argument(
        "-m",
        "--modelnames",
        help="list of model names",
        action="store",
        type=str,
        default="albert-base-v2,roberta-base",
        required=False)
    parser.add_argument(
        "-t",
        "--traincontlevel",
        help="level of train contention",
        action="store",
        type=float,
        default=0.681,
        required=False)

    parser.add_argument(
        "-s",
        "--seedrange",
        help="range of seeds",
        action="store",
        type=list,
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        required=False)


    args = parser.parse_args()
    df_labels = pd.read_csv(args.dataset)
    # Get a single label per instance
    df_labels = df_labels.groupby('text').mean().reset_index()

    # this is a common column name in all data
    df_cont = df_labels.groupby('imgname').mean().reset_index()
    df_cont['contention'] = 'Low'
    df_cont.loc[(df_cont.normative0 >= 0.2) & (
        df_cont.normative0 <= 0.8), 'contention'] = 'High'

    cat = args.attribute_category
    for modelname in args.modelnames.split(','):
        df_descr = get_csv_cross_seeds(train_contention_level=args.traincontlevel, 
            seedrange=args.seedrange,modelname=modelname, cross=1, root_dir=args.root_dir,
            batch_size=batch_size_dict[modelname][1],df_labels=df_labels)
        df_norm = get_csv_cross_seeds(train_contention_level=args.traincontlevel, 
            seedrange=args.seedrange,modelname=modelname, cross=0, root_dir=args.root_dir,
            batch_size=batch_size_dict[modelname][0],df_labels=df_labels)
        df_content = compute_metrics_all(
            df_norm, df_descr, args.seedrange,
            args.traincontlevel)
        df_content.to_csv(
            'cross_dataset_results/text_pretrained/text_auc_{}_{}.csv'.format(
                modelname,
                args.root_dir.split('/')[-2]
            ), index=False)
        
