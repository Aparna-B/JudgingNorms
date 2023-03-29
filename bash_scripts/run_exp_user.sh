#!/bin/bash

set -e
cd ../

slurm_pre="--partition t4v2,rtx6000 --gres gpu:1 --mem 8gb -c 1 --job-name ${1} --output /scratch/ssd001/home/user/github_dir/logs/${1}_%A.log"

output_root="/scratch/ssd001/home/user/github_dir/output"

python sweep.py launch \
    --experiment_name ${1} \
    --output_root "${output_root}" \
    --slurm_pre "${slurm_pre}" \
    --command_launcher "slurm"
