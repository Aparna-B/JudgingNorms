import argparse
import pandas as pd
import numpy as np
from scipy.stats import kruskal, wilcoxon


def compute_mean_sig(dataset, condition1,
                     condition2, category,
                     verbose=False):
    """Get proportion of images significantly different between conditions,
    wrt contention_def group.
    
    Parameters:
      dataset: str, csv file path to dataset (e.g., for the Clothing dataset).
      condition1: str, either descriptive, normative, or context
      condition2: str, either descriptive, normative, or context
      category: int, 0: OR of labels, 1/2/3: factual features.
    
    Returns:
      4 floats, results of Kruskal Wallis H-test and Wilcoxon signed rank test.

    """
    # reading in the dataset
    df_group1 = pd.read_csv(dataset.format(condition1))
    df_group1 = df_group1.groupby('imgname').mean().reset_index()
    df_group2 = pd.read_csv(dataset.format(condition2))
    df_group2 = df_group2.groupby('imgname').mean().reset_index()

    df_group1 = df_group1.sort_values('imgname')
    df_group2 = df_group2.sort_values('imgname')

    assert (df_group1.imgname == df_group2.imgname).all()

    # Statistical test to test if mean label is significantly different (unpaired).
    kruskal_stats, kruskal_p = kruskal(
        df_group1['{}{}'.format(condition1, category)].values,
        df_group2['{}{}'.format(condition2, category)].values)
    
    # Statistical test to test if mean label is significantly different.
    # (paired, one-sided).
    wilcoxon_stats, wilcoxon_p = wilcoxon(
        df_group1['{}{}'.format(condition1, category)].values,
        df_group2['{}{}'.format(condition2, category)].values,
        alternative='greater')

    return (kruskal_stats, kruskal_p,
            wilcoxon_stats, wilcoxon_p)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute significant differences between labels.")
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


    args = parser.parse_args()

    dataset_results = []
    datasets = ['data_dir/dress/{}_labels.csv',
                'data_dir/meal/{}_labels.csv',
                'data_dir/pet/{}_labels.csv',
                'data_dir/toxicity/{}_labels.csv']
    dataset_name = []
    for dataset in datasets:
        result = compute_mean_sig(dataset, args.group1,
                                  args.group2, args.attribute_category,
                                  verbose=False)
        dataset_results.append(result)
        print(dataset)
        dataset_name.append(dataset)

    dataset_results = np.array(dataset_results)
    df_result = pd.DataFrame(
        {
            'Kruskal stat': dataset_results[:, 0],
            'Kruskal pval': dataset_results[:, 1],
            'Wilcoxon stat': dataset_results[:, 2],
            'Wilcoxon pval': dataset_results[:, 3],
            'dataset': dataset_name
        }
    )
    df_result.to_csv('data_dir/{}_{}_stat_tests.csv'.format(
        args.group1,
        args.group2
    ), index=False)

