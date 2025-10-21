
## Harnessing Input-Adaptive Inference for Efficient VLN - Standard VLN Experiments

This directory contains the code for reproducing our standard VLN results on [HAMT](https://arxiv.org/abs/2110.13309) and [DUET](https://arxiv.org/abs/2202.11742).

&nbsp;

---

## Prerequisites

1. Follow the installation instructions provided in the [HAMT](https://github.com/cshizhe/VLN-HAMT/tree/c8b9ee12125f9fe36c51d2ab928fde38f7d846bd) repository to install [Matterport3D](https://github.com/peteanderson80/Matterport3DSimulator), set up the environment, and download the necessary data. **To reproduce our results, you only need to do steps 1–3**. Store the data in `HAMT/datasets`.

**Important:** After installing Matterport3D, set the path to the python build and the directory containing the environment scans in the `.env` file.

2. Follow the installation instructions provided in the [DUET](https://github.com/cshizhe/VLN-DUET) repository to download the necessary data. **You only need to do steps 2–4**. Store the data in `DUET/datasets`.

3. Install the base ViT weights for DUET:

```bash
cd DUET/datasets/R2R/trained_models
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
```

4. Install other dependencies:

```bash
pip install git+https://github.com/ztcoalson/thop.git
pip install dotenv
```

5. Create the panorama images:

```bash
cd HAMT/preprocess
python build_image_lmdb.py
```

Note: By default, the images will be stored at: `Standard-VLN/panos/clean/panoimages.lmdb`. Set the global path in the `.env` file for future experiments.

&nbsp;

---

## Running Our Input-Adaptive Inference Method

To run our input-adaptive inference methods, execute the following commands:

```bash
# HAMT
cd HAMT/finetune_src
scripts/run_{r2r/r2r_back/r2r_last/reverie/cvdn}.sh

# DUET
cd DUET/map_nav_src
scripts/run_{r2r/reverie/soon}.sh
```

To experiment with different configurations, you can vary the following flags:
* `--mode {baseline/mue/efficient/k_only/ee_only/lsh_only/k_lsh/k_ee/ee_lsh}`
    - `baseline`: Use baseline ViT without any adaptive strategies
    - `mue`: Use the [MuE](https://arxiv.org/abs/2211.11152) early-exit strategy
    - `efficient`: Runs our input-adaptive inference method with all three of our proposed strategies (k-extensions, LSH, and early-exit)
    - `k_only`: k-extensions only
    - `ee_only`: early-exit only
    - `lsh_only`: LSH only
    - `k_lsh`: k-extensions and LSH
    - `k_ee`: k-extensions and early-exit
    - `ee_lsh`: early-exit and k-extensions
* `--k`: number of extended views for k-extensions
* `--ee_decay`: decay term for setting the early-exit threshold ("aggressiveness" in the paper)
* `--lsh_threshold`: similarity threshold for re-using views with LSH

Results will be saved to `{HAMT/DUET}/results/{dataset}/{save_id}`. Within the `logs` directory, you can access information about the GFLOPs used per trajectory in `gflops_per_traj_log.txt` and the final results in `valid.txt`.

&nbsp;

---

## Using Different Similarity Metrics

For implementing different similarity metrics, we used the following resources:

- For SSIM and FSIM, we referred to [this package](https://pypi.org/project/image-similarity-measures/).
- For the LPIPs implementation, we referred to [this repository](https://github.com/richzhang/PerceptualSimilarity).

If you would like to change the similarity metric used by LSH, refer to the `LSH` class in `HAMT/finetune_src/r2r/vision_transformer.py` (and `DUET/map_nav_src/r2r/vision_transformer.py`).

&nbsp;

---

## Navigation Under Visual Corruptions

First, create a corrupted `panoimages.lmdb` by running the following command:

```bash
cd HAMT/preprocess/visual_corruption
python build_image_lmdb_degradation_script.py --visual_degradation={'lighting'/'motion_blur'/'speckle_noise'/'spatter'/'defocus_blur'}
```

By default, it will save to `panos/{visual_degradation}/panoimages.lmdb`. To use the corrupted panoramas, replace the `pano_path` variable in `run_{dataset}.sh` with `panos/{visual_degradation}/panoimages.lmdb`.

**Note:** To use the `motion_blur` corruption, you must additionally install `ImageMagick` and the API binding `wand`. If you want to use the other corruptions without installing these packages, you can comment out all `wand` imports in `image_degradations.py`.