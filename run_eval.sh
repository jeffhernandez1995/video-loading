#!/bin/bash

# Use Python to determine the number of CPUs available to this task
NUMBA_DEFAULT_NUM_THREADS=$(python -c "import os; print(len(os.sched_getaffinity(0)))")
export NUMBA_DEFAULT_NUM_THREADS

DATA_PATH=/home/runai-home/data/ffcv/k400/k400_val_16x4_1x5.ffcv
# MODEL=MCG-NJU/videomae-small-finetuned-kinetics

# python eval_model.py \
#     --cfg.dataset $DATA_PATH \
#     --cfg.model $MODEL \
#     --cfg.num_workers 40 \
#     --cfg.batch_size 64 \
#     --cfg.num_chunks 0 \
#     --cfg.chunk_idx 0

# MODEL=MCG-NJU/videomae-base-finetuned-kinetics

# python eval_model.py \
#     --cfg.dataset $DATA_PATH \
#     --cfg.model $MODEL \
#     --cfg.num_workers 40 \
#     --cfg.batch_size 64 \
#     --cfg.num_chunks 0 \
#     --cfg.chunk_idx 0

# MODEL=MCG-NJU/videomae-large-finetuned-kinetics

# python eval_model.py \
#     --cfg.dataset $DATA_PATH \
#     --cfg.model $MODEL \
#     --cfg.num_workers 40 \
#     --cfg.batch_size 64 \
#     --cfg.num_chunks 0 \
#     --cfg.chunk_idx 0

# MODEL=MCG-NJU/videomae-huge-finetuned-kinetics

# python eval_model.py \
#     --cfg.dataset $DATA_PATH \
#     --cfg.model $MODEL \
#     --cfg.num_workers 40 \
#     --cfg.batch_size 32 \
#     --cfg.num_chunks 0 \
#     --cfg.chunk_idx 0

MODEL=facebook/timesformer-base-finetuned-k400

python eval_model.py \
    --cfg.dataset $DATA_PATH \
    --cfg.model $MODEL \
    --cfg.num_workers 40 \
    --cfg.batch_size 64 \
    --cfg.num_chunks 0 \
    --cfg.chunk_idx 0

MODEL=facebook/timesformer-hr-finetuned-k400

python eval_model.py \
    --cfg.dataset $DATA_PATH \
    --cfg.model $MODEL \
    --cfg.num_workers 40 \
    --cfg.batch_size 32 \
    --cfg.num_chunks 0 \
    --cfg.chunk_idx 0

MODEL=google/vivit-b-16x2-kinetics400

python eval_model.py \
    --cfg.dataset $DATA_PATH \
    --cfg.model $MODEL \
    --cfg.num_workers 40 \
    --cfg.batch_size 64 \
    --cfg.num_chunks 0 \
    --cfg.chunk_idx 0

