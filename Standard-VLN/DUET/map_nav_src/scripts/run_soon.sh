#!/bin/bash

##### Important FLAGS #####
outdir="../results/SOON/ours"
pano_path="../../panos/clean/panoimages.lmdb"

mode="efficient" # efficient, mue, baseline, k_only, ee_only, lsh_only, k_lsh, k_ee, ee_lsh

k=4
ee_decay=9e-4
lsh_threshold=0.85
###########################

vit_path="../datasets/R2R/trained_models/jx_vit_base_p16_224-80ecf9dd.pth"

ngpus=1
seed=0

flag="--root_dir ../datasets
      
      --img_db_file ${pano_path}
      --vit_path ${vit_path}
      
      --dataset soon
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert

      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints

      --dagger_sample sample

      --train_alg dagger
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 20
      --max_instr_len 100
      --max_objects 100

      --batch_size 1
      --lr 1e-5
      --iters 200000
      --log_every 1000
      --optim adamW

      --features vitbase
      --obj_features butd
      --image_feat_size 768
      --angle_feat_size 4
      --obj_feat_size 2048

      --ml_weight 0.2   

      --feat_dropout 0.4
      --dropout 0.5

      --mode ${mode}
      --k_ext ${k}
      --ee_decay ${ee_decay}
      --lsh_threshold ${lsh_threshold}

      --gamma 0."

python ./soon/main.py $flag  \
      --tokenizer bert \
      --resume_file ../datasets/SOON/trained_models/best_val_unseen_house \
      --test --submit