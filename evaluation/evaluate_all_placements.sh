#!/bin/bash

if [[ $# -lt 2 ]]; then
echo "Usage $0 <outprefix> [benchmark files]"
exit 1
fi

if [[ -f $1 ]]; then
echo "output prefix $1 is a file --- is the out prefix missing? "
exit 1
fi

number=1000
cgra_setup=benchmarks/architectures/4x4.json
outprefix=$1
shift

while [[ $# -gt 0 ]]; do
    if [[ ! -f $bfile ]]; then
        echo "File $bfile not a file!  Incorrect usage!"
    fi
    bfile=$1
    sbatch ./evaluation/evaluate_placement_slurm.sh PL $number $cgra_setup $bfile exponential $outprefix$(basename $bfile)_out
    shift 1
done