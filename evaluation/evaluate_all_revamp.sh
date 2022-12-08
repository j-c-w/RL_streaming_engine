#!/bin/zsh

typeset -a use_egraphs
zparseopts -D -E -use-egraphs=use_egraphs

if [[ $# -lt 1 ]]; then
echo "Usage $0 [benchmark files]"
exit 1
fi

extra_flags=""
if [[ ${extra_flags[@]} -eq 0 ]]; then
    extra_flags="--use-binary-egraphs"
fi

cgra_setup=benchmarks/architectures/4x4.json

while [[ $# -gt 0 ]]; do
    bfile=$1
    sbatch ./evaluation/evaluate_revamp.sh --slurm $cgra_setup $bfile revamp_output/$(basename $bfile)_out $use_egraphs
    shift
done