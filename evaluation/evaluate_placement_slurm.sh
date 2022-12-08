#!/bin/bash
#SBATCH --cpus-per-task 40
#SBATCH --mem=600GB
#SBATCH -t 3-0:0

set -x
source /etc/profile.d/modules.sh
module load cuda
module load anaconda3
conda init bash
source ~/.bashrc
conda activate streaming_engine
export PATH=$PATH:~/Projects/CGRA/CGRA-Mapper/bin
export PYTHON_PATH=$PWD

./evaluation/evaluate_placement.sh $@