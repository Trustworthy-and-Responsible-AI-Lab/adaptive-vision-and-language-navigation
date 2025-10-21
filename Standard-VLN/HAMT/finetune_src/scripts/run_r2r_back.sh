#!/bin/bash

##### Important FLAGS #####
outdir="../results/R2R_BACK/ours"
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
      
      --dataset r2r_back

      --output_dir ${outdir}
      --img_db_file ${pano_path}
      --vit_path ${vit_path}

      --seed ${seed}

      --ngpus ${ngpus}

      --fix_lang_embedding
      --fix_hist_embedding

      --hist_enc_pano
      --hist_pano_num_layers 2
      
      --features vitbase_r2rfte2e
      --feedback sample

      --maxAction 30
      --batch_size 1
      --image_feat_size 768

      --lr 1e-5
      --iters 300000
      --log_every 1000
      --optim adamW

      --mlWeight 0.2
      --maxInput 60
      --angle_feat_size 4
      --featdropout 0.4
      --dropout 0.5

      --mode ${mode}
      --k_ext ${k}
      --ee_decay ${ee_decay}
      --lsh_threshold ${lsh_threshold}"

python ./r2r/main.py $flag \
      --resume_file ../datasets/R2R/trained_models/vitbase-finetune-e2e/best_val_unseen \
      --test --submit