#!/bin/bash

set -x
if [[ $# -lt 1 ]]; then
	echo "Usage: $0 <outfiles>"
	exit 1
fi

set -eu

typeset -a rw_files
i=0
while [[ $# -gt 0 ]]; do
	grep -e 'total reward' $1 > .total_reward_${1}
	rw_files+=( .total_reward_${1} )
	i=$(( i + 1 ))
	shift
done
set -x
python plot_rewards.py ${rw_files[@]}
