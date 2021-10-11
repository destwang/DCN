#!/bin/bash

set -v
set -e


INPUT_FILE=data/sighan15/TestInput.txt
OUTPUT_FILE=output.txt
MODEL_DIR=dcn_models/checkpoint-8794
MAX_LENGTH=130
BATCH_SIZE=4

python predict_DCN.py \
    --model $MODEL_DIR \
    --input_file $INPUT_FILE \
    --output_file $OUTPUT_FILE \
    --batch_size $BATCH_SIZE \
	--max_len $MAX_LENGTH
