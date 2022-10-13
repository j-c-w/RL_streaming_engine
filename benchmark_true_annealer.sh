#!/bin/zsh

typeset -a plot_only
zparseopts -D -E -plot-only=plot_only

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <number of seeds to try>"
	exit 1
fi

if [[ ${#plot_only} -eq 0 ]]; then
	if [[ -d true_annealer_output ]]; then
		rm -rf true_annealer_output.old
		mv true_annealer_output true_annealer_output.old
	fi
	mkdir -p true_annealer_output

	parallel -j10 "
	python sim_anneal_train.py --input input_graphs/ifft_inner_loop_ir.json --device-topology 16 6 --true-annealer --seed {} &> true_annealer_output/{}.out
	" ::: $(seq 0 $1)
fi

res_file=true_annealer_output/rfile
echo -n "" > $res_file
for file in $(find true_annealer_output/ -name "*.out"); do
	tail -n 2 $file | head -n 1 >> $res_file
done

python plot_annealing_results.py $res_file data_files/ml_assisted_annealer
