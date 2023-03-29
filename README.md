# Judging Facts, Judging Norms: Training Machine Learning Models to Judge Humans Requires a New Approach to Labeling Data <!-- omit in toc -->
An examination of data labeling practices for normative applications. 

## Contents <!-- omit in toc -->
- [Setting Up](#setting-up)
  - [1. Environment and Prerequisites](#1-environment-and-prerequisites)
  - [2. Obtaining the Data](#2-obtaining-the-data)
- [Main Experimental Grid](#main-experimental-grid)
  - [1. Running Experiments](#1-running-experiments)
  - [2. Aggregating Results](#2-aggregating-results)
- [Data and Model Sheets](#data-and-model-sheets)
  - [Data Sheet](#data-and-model-sheets)
  - [Model Sheet](#data-and-model-sheets)
- [Citation](#citation)


## Setting Up
### 1. Environment and Prerequisites
Run the following commands to clone this repo and create the Conda environment:

```
git clone repo
cd repo/
conda env create -f environment.yml
conda activate label_exp
```

### 2. Obtaining the Data
We provide the `Clothing`, `Meal`, `Pet`, and `Comment` datasets as .csv files in this repository. 


## Main Experimental Grid
### 1. Running Experiments
To reproduce the experiments in the paper which involve training grids of models using different hyperparameters, refer to files within the `image_models` and `text_models` folders.

```
bash ${folder}/{bash_script} 
```

where:
- `folder` corresponds to either `image_models` or `text_models`
- `bash_script` corresponds to script used on the compute cluster 

Sample bash scripts showing the command can also be found in `bash_scripts/`.
Jobs can also be launched using the `sweep.py` in `image_models` as:

```
python sweep.py launch \
    --experiment_name {experiment_name} \
    --output_dir {output_root} \
    --command_launcher {launcher} 
```

### 2. Aggregating Results
We aggregate results and generate tables using the aggregation scripts in `lib`.

## Data and Model Sheets
Data and model sheets for all analyses can be found in the `data_model_sheets` folder.

## Citation
If you use this code in your research, please cite the corresponding publication.
