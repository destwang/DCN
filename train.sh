#!/bin/bash

set -v
set -e


TRAIN_FILE=data/train.txt
TEST_FILE=data/sighan15/test_format.txt
BERT_MODEL=chinese_roberta_wwm_ext_pytorch/
OUTPUT_DIR=dcn_models/
SAVE_STEPS=8794
SEED=1038
LR=5e-5
SAVE_TOTAL_LIMIT=5
MAX_LENGTH=130
BATCH_SIZE=4
NUM_EPOCHS=10

python train_DCN.py \
    --output_dir $OUTPUT_DIR \
	--learning_rate $LR  \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --model_type=bert \
    --model_name_or_path=$BERT_MODEL \
    --num_train_epochs $NUM_EPOCHS \
    --save_steps $SAVE_STEPS \
	--logging_steps $SAVE_STEPS \
	--save_total_limit $SAVE_TOTAL_LIMIT \
	--block_size $MAX_LENGTH \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --do_train \
    --do_eval \
    --do_predict \
	--evaluate_during_training \
    --seed $SEED \
    --mlm \
	--mlm_probability 0.15
