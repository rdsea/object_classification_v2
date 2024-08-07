#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <tflite-directory> <output-directory>"
  exit 1
fi

tflite_dir="$1"
output_dir="$2"

if [ ! -d "$tflite_dir" ]; then
  echo "Error: Input directory $tflite_dir does not exist."
  exit 1
fi

# Check if the output directory exists, create it if it does not
if [ ! -d "$output_dir" ]; then
  echo "Output directory $output_dir does not exist. Creating it."
  mkdir -p "$output_dir"
fi

for tflite_file in "$tflite_dir"/*.tflite; do
  if [ ! -e "$tflite_file" ]; then
    echo "No .tflite files found in the input directory $tflite_dir."
    exit 1
  fi

  base_name=$(basename "$tflite_file" .tflite)

  onnx_file="${output_dir}/${base_name}.onnx"

  if python -m tf2onnx.convert --tflite "$tflite_file" --output "$onnx_file"; then
    echo "Successfully converted $tflite_file to $onnx_file"
  else
    echo "Failed to convert $tflite_file"
  fi
done
