#!/bin/zsh
#SBATCH --cpus-per-task 40
#SBATCH --mem=600GB
#SBATCH -t 3-0:0

typeset -a annealer dse slurm
zparseopts -D -E -- -annealer=annealer -dse=dse -slurm=slurm


if [[ $# -ne 4 ]]; then 
	echo "Usage: $0 <model> <cgra> <benchmarks> <output_folder>"
	exit 1
fi

if [[ ${#slurm[@]} -gt 0 ]]; then
	echo "Loading module"
	source /etc/profile.d/modules.sh
	module load cuda/11.4
	source ~/.bashrc
	export PATH=$PATH:~/Projects/CGRA/CGRA-Mapper/bin
	conda activate streaming_engine
fi

model=$1
cgra=$2
benchmarks=$3
output_folder=$4

if [[ -d output_eval/$output_folder ]]; then
	rm -rf output_eval/$output_folder
fi
mkdir -p output_eval/$output_folder

echo "Output folder is output_eval/$output_folder"

python cgra_place.py $cgra $benchmarks exponential --random-cgras 1000 --restore $model --manual-distribution --model-no-cgra-state --use-egraphs --temp-folder temp_folder_evaluation  --test > output_eval/$output_folder/testing