#!/bin/bash

python entry.py\
  --batch_size 128 \
  --acc_batches 1 \
  --d_model 256 \
  --dim_feedforward 512 \
  --epochs 2000 \
  --dropout 0.2 \
  --warmup_updates 2000 \
  --tot_updates 1000000 \
  --dataset data/uspto \
  --known_rxn_cnt \
  --norm_first \
  --nhead 32 \
  --p_layer 4 \
  --r_layer 4 \
  --output_layer 4 \
  --seed 123 \
  --cuda 1 \
  --max_single_hop 4 \
  --log_dir debug \
  --not_fast_read