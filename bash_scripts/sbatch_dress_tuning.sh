#!/bin/bash

# xargs hyperparameters
max_proc=180
partition='gpu' # cpu or gpu

# command file
cmdfile=justification.txt

# hyperparameters
batch_sizes="32 64 128"
learning_rates="0.1 0.01 0.001"
weightdecays="0.1 0.5"


# input parameters
experiments="descriptive normative"
category="0"
seeds="1 2 3 4 5"
train="1"
cross="0"
transfer="0"
contention="0.242"

noise="0"
sizes="1"
dataset_name='dress'
data_root="datasets"
img_root='dress_images'

# loop over variables and echo commands
for weightdecay in $weightdecays
do
    for experiment in $experiments
    do
        for batch_size in $batch_sizes
        do
            for learning_rate in $learning_rates
            do
                for seed in $seeds
                do
              	    cmd="python run.py --model_name=resnet50_all --contention_ref=normative --contention=${contention} --batch_size=${batch_size} --learning_rate=${learning_rate} --weight_decay=${weightdecay} --label_noise=${noise} --csv_file {}_labels.csv --data_root=${data_root} --img_root=${img_root} --dataset_name=${dataset_name} --logfile=logs/latest/${dataset_name}_${experiment}_contention_${contention}_category_${category}_seed_${seed}_cross.log --seed=${seed} --experiment=${experiment} --category=${category} --train=1 --cross=0 --transfer=0"                    
		    echo $cmd
                done
            done
        done
    done
done > $cmdfile

# dispatch jobs with xargs
echo "Sample command: " $cmd
cmd=( $cmd )
num_tokens=${#cmd[@]}
echo "Number of tokens: " $num_tokens
if [ $partition == 'cpu' ]; then
    xargs -n $num_tokens -P $max_proc srun --mem=16G -p cpu < $cmdfile
else
    xargs -n $num_tokens -P $max_proc srun --gres=gpu:1 -p t4v2,rtx6000 --mem=16G < $cmdfile
fi

