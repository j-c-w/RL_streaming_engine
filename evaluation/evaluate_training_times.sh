#!/bin/zsh

typeset -a slurm

if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--slurm]"
    echo "This script submits evaluation jobs for all the different sizes of script"
    exit 1
fi

