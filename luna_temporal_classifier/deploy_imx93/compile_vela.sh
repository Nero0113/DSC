#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 MODEL_DIR OUTPUT_DIR" >&2
    exit 2
fi

model_dir=$1
output_dir=$2
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$output_dir"

cp "$model_dir/manifest.json" "$output_dir/manifest.json"
cp "$model_dir/preprocessing.npz" "$output_dir/preprocessing.npz"
cp "$model_dir/inception_encoder_int8.tflite" "$output_dir/inception_encoder_int8.tflite"
cp "$model_dir/context_tcn_int8.tflite" "$output_dir/context_tcn_int8.tflite"
cp "$script_dir/runtime_imx93.py" "$output_dir/runtime_imx93.py"

vela \
    --accelerator-config ethos-u65-256 \
    --output-dir "$output_dir" \
    "$model_dir/inception_encoder_int8.tflite"

vela \
    --accelerator-config ethos-u65-256 \
    --output-dir "$output_dir" \
    "$model_dir/context_tcn_int8.tflite"

echo "Vela models written to $output_dir"
