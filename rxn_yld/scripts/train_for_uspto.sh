#!/bin/bash

python entry.py\
  --batch_size 256 \
  --acc_batches 1 \
  --d_model 128 \
  --dim_feedforward 256 \
  --epochs 2000 \
  --dropout 0.2 \
  --warmup_updates 200000 \
  --tot_updates 100000000 \
  --dataset data/uspto \
  --known_rxn_cnt \
  --norm_first \
  --nhead 8 \
  --p_layer 3 \
  --r_layer 3 \
  --output_layer 3 \
  --seed 123 \
  --cuda 1,2 \
  --max_single_hop 4 \
  --log_dir tb_lgs \
  --not_fast_read
