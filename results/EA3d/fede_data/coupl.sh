#!/bin/bash

input_folder="$1"
output_folder="$2"
export LC_NUMERIC="en_US.UTF-8"

for input_file in "$input_folder"/*.dat; do
    output_file="${output_folder}/$(basename "$input_file" .dat)_output.txt"
    awk '{ print $1-1, $2-1, $3 * 1e-7 }' "$input_file" > "$output_file"
done
