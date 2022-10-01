#!/bin/bash

set -eu
if [[ $# -ne 6 ]]; then
	echo "Usage: $0 <mode> <number of architectures> <cgra specifiation>  <benchmark description.json> <exploration mode> <output folder>"
	echo "Mode should be RL (Reinforcement learning) or SA (Simulated Annealing)"
	echo "Number of Arechitectures is the number of architectures to explore. "
	echo ' CGRA specification needs to have 
	{
		"rows": ...,
		"cols": ...,
		"operations": ["..."]
	}

	where operations is a list of operations to be selected from using
	the exploration model.
'
	echo 'benchmark desciption should be a json of the format:
	{
		"benchmarks":
		[
		{ "file": ...,
			"weight": <scaling factor of importance>
		}
		],
	}
	'
	echo "explotaiton mode should be something tat the cgra_place tool accepts"
	exit 1
fi

mode=$1
number=$2
output_folder=$6
shift 2

mkdir -p temp_eval
mkdir -p output_eval/$output_folder

echo "Mode is $mode"
if [[ $mode == *RL* ]]; then
	echo "Running RL "
	python cgra_place.py --print-cgras --model-no-cgra-state --temp-folder temp_eval $1 $2 $3 --random-cgras $number --train-only --save-model trained_model.pk > output_eval/$output_folder/training
	# This uses the same CGRAs becuase it uses the same random seed to geenrate them.
	python cgra_place.py --model-no-cgra-state --temp-folder temp_eval $1 $2 $3 --random-cgras $number --restore trained_model.pk --test > output_eval/$output_folder/testing
else
	# Do everything in a singl placement using simulated annealing.
	echo "Running SA"
	python cgra_place.py --temp-folder evaluation_temps $1 $2 $3 --random-cgras $number --annealer > output_eval/$output_folder/annealing
fi
