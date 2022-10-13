#!/bin/bash

if [[ $# -lt 1 ]]; then
echo "Usage $0 [benchmark files]"
exit 1
fi

cgra_setup=benchmarks/architectures/4x4.json

while [[ $# -gt 0 ]]; do
    bfile=$1
    sbatch ./evaluation/evaluate_revamp.sh --slurm $cgra_setup $bfile revamp/$(basename $bfile)_out
    shift
done