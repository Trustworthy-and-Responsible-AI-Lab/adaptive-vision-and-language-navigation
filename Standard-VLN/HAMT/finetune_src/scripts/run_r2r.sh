#!/bin/bash

##### Important FLAGS #####
outdir="../results/R2R/ours"
pano_path="../../panos/clean/panoimages.lmdb"

mode="efficient" # efficient, mue, baseline, k_only, ee_only, lsh_only, k_lsh, k_ee, ee_lsh

k=4
ee_decay=9e-4
lsh_threshold=0.85
###########################

vit_path="../datasets/R2R/trained_models/vit_step_22000.pt"

ngpus=1
seed=0

flag="--root_dir ../datasets
      --output_dir ${outdir}

      --dataset r2r

      --img_db_file ${pano_path}
      --vit_path ${vit_path}

      --vlnbert ${vlnbert}
      --ob_type pano
      
      --world_size ${ngpus}
      --seed ${seed}
      
      --num_l_layers 9
      --num_x_layers 4

      --hist_enc_pano

      --fix_lang_embedding
      --fix_hist_embedding

      --features vitbase_r2rfte2e
      --feedback argmax

      --max_action_len 15
      --max_instr_len 60

      --image_feat_size 768
      --angle_feat_size 4

      --lr 1e-5
      --iters 300000
      --log_every 2000
      --batch_size 1
      --optim adamW

      --ml_weight 0.2      

      --feat_dropout 0.4
      --dropout 0.5

      --mode ${mode}
      --k_ext ${k}
      --ee_decay ${ee_decay}
      --lsh_threshold ${lsh_threshold}"

python ./r2r/main.py $flag \
      --resume_file ../datasets/R2R/trained_models/vitbase-finetune-e2e/best_val_unseen \
      --test --submit