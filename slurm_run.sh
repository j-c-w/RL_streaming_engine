#!/bin/bash

conda activate streaming_engine
module load cuda/11.4
python sim_anneal_train.py $@
