#!/bin/bash

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 stdout_file"
	exit 1
fi

mkdir -p data_files


tail -n 10000 $1 | grep -B 1 "After iter" | grep -e "Reducing" > data_files/ml_assisted_annealer
