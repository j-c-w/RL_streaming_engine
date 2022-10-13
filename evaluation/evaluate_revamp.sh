#!/bin/zsh
#SBATCH --cpus-per-task 40
#SBATCH --mem=400GB
#SBATCH -t 2-0:0

typeset -a slurm
zparseopts -D -E -- -slurm=slurm

set -eu

if [[ ${#slurm} -gt 0 ]]; then
    # Set up terminal properly for slurm running
    source /etc/profile.d/modules.sh
    module load cuda
    module load anaconda3
    source ~/.bashrc
    conda activate streaming_engine
    export PATH=$PATH:~/Projects/CGRA/CGRA-Mapper-2/bin
    export PYTHON_PATH=$PWD
fi


# This script evaluates the approach for architectural parameter selection used
# by REVAMP.
if [[ $# -ne 3 ]]; then
    echo "Usage: <cgra specification> <benchmark description.json> <output folder>"
    echo "cgra specification needs to have { 'rows': ..., 'cols': ... , 'num_ops': 10 }"
    echo "benchmark description should be as in evaluate placement."

    echo "This script goes through the benchmarks, and uses the methodlogy from Bandara ASPLOS 2022 to generate
    CGRA elements.  It then uses simulated annealing to place those operations."
    exit 1
fi

cgra_spec=$1
bench_desc=$2
output_folder=$3

mkdir -p output_eval/$output_folder

python evaluation/revamp/compute_operations.py $cgra_spec $bench_desc output_eval/$output_folder

# Now that we have generated the architecture, use the SA placer.
python cgra_place.py --temp-folder evaluation_temps $PWD/output_eval/$output_folder/cgra.json $bench_desc none --annealer --manual-distribution > output_eval/$output_folder/annealing