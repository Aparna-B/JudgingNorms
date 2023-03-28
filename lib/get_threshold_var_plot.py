import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from textwrap import wrap

viols = ['descriptive0', 'normative0']
colors = ['blue', 'green', 'red', 'yellow']

datasets = ['data_dir/dress/{}_labels.csv',
            'data_dir/meal/{}_labels.csv',
            'data_dir/pet/{}_labels.csv',
            'data_dir/toxicity/{}_labels.csv']
fig, ax = plt.subplots()
fig.set_size_inches(15, 6)
thresholds = np.arange(20)[1:] / 20
print(thresholds)
colors = ['sienna', 'steelblue', 'darkolivegreen', 'mediumvioletred']

legend_labels = ['Clothing Rule', 'Meal Rule',
                 'Pet Rule', 'Comment Rule']
lines = []
for dataset_num, dataset in enumerate(datasets):
    vari = []
    for threshold in thresholds:
        d = pd.read_csv(dataset.format('descriptive')).groupby(
            'imgname').mean().reset_index().copy()
        n = pd.read_csv(dataset.format('normative')).groupby(
            'imgname').mean().reset_index().copy()

        n['pred'] = n[viols[1]] > threshold
        d['pred'] = d[viols[0]] > threshold
        n = n.sort_values('imgname')
        d = d.sort_values('imgname')
        assert (n.imgname==d.imgname).all()

        k = n.loc[n.pred.values != d.pred.values]
        vari.append(len(k) / d.shape[0])

    axes = sns.lineplot(x=thresholds,
                        y=vari, marker="D",
                        color=colors[dataset_num]
                        )

title_text = "%" + \
    "Examples with Different Normative and Descriptive Label"
text = "%" + "Examples with Different Labels"


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], marker='D', color=color,
                       markerfacecolor=color, markersize=10) for color in colors

                ]

plt.legend(legend_labels,  ncol=4, loc='lower center', fontsize=15)


plt.ylabel('\n'.join(wrap(text, 45
                          )), fontsize=15
           )
plt.title('\n'.join(wrap(title_text, 80
                         )), fontsize=20
          )

plt.xlabel('Threshold', fontsize=15)
plt.xticks(np.arange(20)[1:] / 20, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('FigureApp2.png', dpi=300)
