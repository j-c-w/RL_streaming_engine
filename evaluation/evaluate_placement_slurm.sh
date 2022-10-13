#!/bin/bash
#SBATCH --cpus-per-task 40
#SBATCH --mem=400GB
#SBATCH -t 2-0:0

source /etc/profile.d/modules.sh
module load cuda
module load anaconda3
conda init bash
source ~/.bashrc
conda activate streaming_engine
export PATH=$PATH:~/Projects/CGRA/CGRA-Mapper-2/bin
export PYTHON_PATH=$PWD

./evaluation/evaluate_placement.sh $@