#!/bin/zsh

set -eu

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

python evaluation/revamp/compute_operations.py $cgra_spec $bench_desc output_eval/$output_folder/cgra.json

# Now that we have generated the architecture, use the SA placer.
python cgra_place.py --temp-folder evaluation_temps output_eval/$output_folder/cgra.json $bench_desc none --annealer --manual-distribution > output_eval/$output_folder/annealing