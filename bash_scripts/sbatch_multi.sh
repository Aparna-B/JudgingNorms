#!/bin/bash

# xargs hyperparameters
max_proc=80
partition='gpu' # cpu or gpu

# command file
cmdfile=output.txt

# input parameters
experiments="0 1"
categories="0"
seeds="1 2 3 4 5"
weights="1"
contentions="0.69"
model_names="roberta-base albert-base-v2"
seeds="1 2 3 4 5 6 7 8 9 10"
batch_sizes="10 32"


# loop over variables and echo commands
for model_name in $model_names
do
    for seed in $seeds
    do
	for experiment in $experiments
	do
		for batch_size in $batch_sizes
		do
    			cmd="python train_multi.py --model_name=${model_name} --cross=${experiment} --weight=1 --seed=${seed} --contention=0.681 --batch_size=${batch_size}"
    			echo $cmd
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
    xargs -n $num_tokens -P $max_proc srun --gres=gpu:1 -p t4v1,t4v2,rtx6000 --mem=8G < $cmdfile
fi

