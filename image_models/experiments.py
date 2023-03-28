import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

def combinations(grid):
    return list(dict(zip(grid.keys(), values)) for values in product(*grid.values()))

def get_hparams(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment]().get_hparams()

def get_script_name(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].fname


#### write experiments here
class dress_experiments():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment_name':['dress_experiments'],
            'model_name': ['resnet50_all'],
            'contention_ref':['normative'],
            'contention':[0.242],
            'batch_size':[32,64,128],
            'learning_rate':[0.1,0.01,0.001],
            'weight_decay':[0.1,0.5],
            'label_noise':[0],
            'csv_file':['{}_labels.csv'],
            'data_root':['datasets'],
            'img_root':['dress_images'],
            'dataset_name':['dress'],
            'logfile':['logs/latest/dress_1.log'],
            'seed':[1,2,3,4,5],
            'experiment': ['descriptive','normative'],
            'category':[0],
            'train':[1],
            'cross':[0],
            'transfer':[0]
        }

    def get_hparams(self):
        return combinations(self.hparams)



class meal_experiments():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment_name': ['meal_experiment'],
            'model_name': ['resnet50_all'],
            'contention_ref':['normative'],
            'contention':[0.4],
            'batch_size':[32,64,128],
            'learning_rate':[0.1,0.01,0.001],
            'weight_decay':[0.1,0.5],
            'label_noise':[0],
            'csv_file':['{}_labels.csv'],
            'data_root':['food_dataset'],
            'img_root':['food_images_from_s3'],
            'dataset_name':['meal'],
            'logfile':['logs/latest/meal_1.log'],
            'seed':[1,2,3,4,5],
            'experiment': ['descriptive','normative'],
            'category':[0],
            'train':[1],
            'cross':[0],
            'transfer':[0]
        }

    def get_hparams(self):
        return combinations(self.hparams)


class pet_experiments():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment_name': ['pet_experiment'],
            'model_name': ['resnet50_all'],
            'contention_ref':['normative'],
            'contention':[0.542],
            'batch_size':[32,64,128],
            'learning_rate':[0.1,0.01,0.001],
            'weight_decay':[0.1,0.5],
            'label_noise':[0],
            'csv_file':['{}_labels.csv'],
            'data_root':['dog_dataset'],
            'img_root':['dog_images'],
            'dataset_name':['pet'],
            'logfile':['logs/latest/pet_1.log'],
            'seed':[1,2,3,4,5],
            'experiment': ['descriptive','normative'],
            'category':[0],
            'train':[1],
            'cross':[0],
            'transfer':[0]
        }

    def get_hparams(self):
        return combinations(self.hparams)


