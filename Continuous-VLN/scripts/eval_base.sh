#!/bin/bash

python run.py \
    --exp-config sim2sim_vlnce/config/sgm-local_policy.yaml \
    EVAL_CKPT_PATH_DIR ./data/models/RecVLNBERT-ce_vision-tuned.pth \
    LOG_FILE ./logs/eval_base.log
