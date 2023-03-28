import pandas as pd
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Obtain normative vs descriptive violations.")
    parser.add_argument(
        "-dataset",
        "--dataset",
        help="Dataset csv path",
        action="store",
        type=str,
        default='income',
        required=False)
    args = parser.parse_args()
    dataset=args.dataset
    df_descr = pd.read_csv('data_dir/{}/descriptive_labels.csv'.format(dataset))
    df_norm = pd.read_csv('data_dir/{}/normative_labels.csv'.format(dataset))
    df_context = pd.read_csv('data_dir/{}/context_labels.csv'.format(dataset))

    df_descr = df_descr.groupby('imgname').mean().reset_index()
    df_norm = df_norm.groupby('imgname').mean().reset_index()
    df_context = df_context.groupby('imgname').mean().reset_index()
    norm_prop = df_norm[df_norm['normative0'] > 0.5].shape[0] / df_norm.shape[0]
    descr_prop = df_descr[df_descr['descriptive0'] > 0.5].shape[0] / df_descr.shape[0]
    context_prop= df_context[df_context['context0'] > 0.5].shape[0] / df_context.shape[0]
    print("{} % descriptive violations, {} % normative violations, {} % context_prop".format(
        descr_prop, norm_prop,context_prop))
