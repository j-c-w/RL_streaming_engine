#!/bin/zsh
#SBATCH --cpus-per-task 40
#SBATCH --mem=100GB
#SBATCH --time 72:00:00

if [[ $# -lt 1 ]]; then
	echo "Usage: $0 <Stdout output file> [ args for trainer ]"
	exit 1
fi

outfile=$1
shift

export PATH=$PATH:~/Projects/CGRA/CGRA-Mapper/bin

export CUDA_VISIBLE_DEVICES=""
eval "$(conda shell.zsh hook)"
conda activate streaming_engine
./run_place.sh $@ &> $outfile
