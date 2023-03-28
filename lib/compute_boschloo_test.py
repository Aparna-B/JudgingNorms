import argparse
import pandas as pd
import numpy as np
from scipy.stats import boschloo_exact


def compute_sig_dif_boschloo(dataset, condition1,
                             condition2, category,
                             verbose=False,
                             contention_def=-1):
    """Get proportion of images significantly different between conditions,
    wrt contention_def group.

    Parameters:
      dataset: str, csv file path to dataset (e.g., for the Clothing dataset).
      condition1: str, either descriptive, normative, or context
      condition2: str, either descriptive, normative, or context
      category: int, 0: OR of labels, 1/2/3: factual features.
      contention_def: int, category for defining contention

    """
    # reading in the dataset
    df_group1 = pd.read_csv(dataset.format(condition1))
    df_group2 = pd.read_csv(dataset.format(condition2))
    df_default = pd.read_csv(dataset.format('descriptive'))

    df_sig_diff = []
    sig_diff = []
    total_count = 0
    img_urls = []
    group1_contention = []
    group2_contention = []
    group_default_contention = []

    # NB: group{i} denotes one of the condition here,
    # We use group_default to get significant differences for 
    # high and low contention separately, defined with respect 
    # to a "default" condition. Here, we use "descriptive" as default.
    for image_url in df_group1['imgname'].unique():
        group1 = df_group1[df_group1['imgname'] == image_url][
            condition1 + str(category)].astype(int).values
        group2 = df_group2[df_group2['imgname'] == image_url][
            condition2 + str(category)].astype(int).values
        group_default = df_default[df_default['imgname'] == image_url][
            'descriptive' + str(category)].astype(int).values

        # asserting that only two possible values (0: No, 1:Yes)
        # i.e., assert no empty values
        assert set(group1) in [{0}, {1}, {0, 1}]
        assert set(group2) in [{0}, {1}, {0, 1}]
        assert set(group_default) in [{0}, {1}, {0, 1}]

        res = boschloo_exact([
            [len(group2[group2 == 1]), len(group1[group1 == 1])],
            [len(group2[group2 == 0]), len(group1[group1 == 0])]],
            alternative='two-sided')



        # statistic and pvalue from test
        p1 = res.pvalue
        if np.isnan(p1):
            assert len(set(group1))==1
            assert len(set(group2))==1
            assert set(group1)==set(group2)
            p1=1

        if p1 <= 0.05:
            sig_diff.append(1)
            if verbose:
                print(
                    image_url,
                    np.mean(group1), np.mean(group2))

        else:
            sig_diff.append(0)
        total_count += 1

        # getting contention label
        group1_contention.append(int(0.2 <= np.mean(group1) <= 0.8))
        group2_contention.append(int(0.2 <= np.mean(group2) <= 0.8))
        group_default_contention.append(
            (int(0.2 <= np.mean(group_default) <= 0.8)))
        img_urls.append(image_url)
    sig_diff = np.array(sig_diff)

    df_sig_diff = pd.DataFrame({
        'significant_difference': sig_diff,
        'imgname': img_urls,
        'group1_contention': group1_contention,
        'group2_contention': group2_contention,
        'group_default_contention': group_default_contention
    })

    # computing proportions of images significantly different between
    # conditions
    overall_prop = np.mean(
        df_sig_diff['significant_difference'].values)

    if contention_def == 0:
        df_sig_diff_high = df_sig_diff[df_sig_diff['group1_contention'] == 1]
        df_sig_diff_low = df_sig_diff[df_sig_diff['group1_contention'] == 0]
    elif contention_def == 1:
        df_sig_diff_high = df_sig_diff[df_sig_diff['group2_contention'] == 1]
        df_sig_diff_low = df_sig_diff[df_sig_diff['group2_contention'] == 0]
    else:
        print('here')
        df_sig_diff_high = df_sig_diff[
            df_sig_diff['group_default_contention'] == 1]
        df_sig_diff_low = df_sig_diff[
            df_sig_diff['group_default_contention'] == 0]

    contentious_prop = np.mean(
        df_sig_diff_high['significant_difference'].values)
    uncontentious_prop = np.mean(
        df_sig_diff_low['significant_difference'].values)

    return overall_prop, contentious_prop, uncontentious_prop


def get_all_diff(group1='descriptive', group2='normative'):
    p1, p2, p3 = [], [], []
    for dataset in ['dress', 'meal', 'pet', 'toxicity']:
        overall_prop, contentious_prop,\
            uncontentious_prop = compute_sig_dif_boschloo(
                dataset='data_dir/' + dataset + '/{}_labels.csv',
                condition1=group1,
                condition2=group2,
                category=0,
                verbose=False)
        p1.append(overall_prop)
        p2.append(contentious_prop)
        p3.append(uncontentious_prop)
    pd.DataFrame({
        'dataset': ['dress', 'meal', 'pet', 'toxicity'],
        'overall': p1,
        'contentious': p2,
        'uncontentious': p3
    }).to_csv('all_datasets_boschloo_{}_{}_test.csv'.format(group1,
                                                            group2), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute significant differences between labels.")
    parser.add_argument(
        "-d",
        "--dataset",
        help="path to csv with descriptive, normative, and context labels",
        action="store",
        type=str,
        default='data_dir/toxicity/{}_labels.csv',
        required=False)
    parser.add_argument(
        "-g1",
        "--group1",
        help="first labelling condition in stat test",
        action="store",
        type=str,
        default='descriptive',
        required=False)
    parser.add_argument(
        "-g2",
        "--group2",
        help="second labelling condition in stat test",
        action="store",
        type=str,
        default='normative',
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
        "--contention_def",
        help="labelling condition to get contention definition for"
        "0: group1, 1: group2, -1: defaults to descriptive",
        action="store",
        type=int,
        default=-1,
        required=False)

    parser.add_argument(
        "--all_run",
        help="Flag to run for all datasets",
        action="store",
        type=int,
        default=1,
        required=False)

    args = parser.parse_args()

    if args.all_run:
        get_all_diff('descriptive', 'normative')
        get_all_diff('context', 'normative')
        get_all_diff('descriptive', 'context')

    else:

        overall_prop, contentious_prop,\
            uncontentious_prop = compute_sig_dif_boschloo(
                dataset=args.dataset,
                condition1=args.group1,
                condition2=args.group2,
                category=args.attribute_category,
                verbose=False,
                contention_def=args.contention_def)
        print(
            "overall_prop: {}%, contentious_prop: {}% "
            ",uncontentious_prop: {}%".format(
                overall_prop, contentious_prop,
                uncontentious_prop,
            ))
