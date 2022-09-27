#!/bin/bash

files=( loop_ffmpeg_66.c loop_ffmpeg_327.c loop_freeimage_665.c )
typeset -a full_files
for file in ${files[@]}; do
    full_files+=( ~/Projects/CGRA/CGRA-Mapper/Loops4/$file )
done
python cgra_place.py ~/Projects/CGRA/CGRA-Mapper/benchmark_scripts/architectures/cca.json ${full_files[@]} $@
