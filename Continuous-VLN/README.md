## Harnessing Input-Adaptive Inference for Efficient VLN - Continuous VLN Experiments

This directory contains the code for reproducing our continuous VLN results on [VLN-CE-BERT](https://arxiv.org/abs/2204.09667).

&nbsp;

---

## Prerequisites

1. Set up the environment:

```bash
### Manually install submodules

# transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 067923d3267325f525f4e46f357360c191ba562e && cd ..

# habitat-lab
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout d6ed1c0a0e786f16f261de2beafe347f4186d0d8 && cd ..

### Create a conda environment with preliminary dependencies
conda create -n vln-ce python=3.6 caffe-gpu
conda activate vln-ce

### Install `habitat-sim=0.1.7` (with `headless`)
conda install -c aihabitat habitat-sim=0.1.7=*headless* -c conda-forge
```

**Important:** Before installing pip dependencies, you may need to change instances of `python-opencv>=3.3.0` to `opencv-python==3.4.0.12` in `./habitat-lab/requirements.txt`. Without this, pip tries to install later versions of the package that require wheels, which led to compilation errors for us.

```bash
### Install the remaining pip dependencies
pip install -r habitat-lab/requirements.txt
pip install -r habitat-lab/habitat_baselines/rl/requirements.txt
pip install -r habitat-lab/habitat_baselines/rl/ddppo/requirements.txt
pip install -r habitat-lab/habitat_baselines/il/requirements.txt
pip install -r requirements.txt
pip install -e transformers
pip install -e habitat-lab

### (Optional) You may need to uninstall libegl and libgl; they caused OpenGL errors for us
conda uninstall libegl libgl
```

2. Download model files for our scan-only SGM:

```bash
cd data
gdown "https://drive.google.com/uc?id=1HOd1rk2vKiNZ9PqJGt_pA5SfFbUm5bcv"
unzip efficient_sgm_models.zip && cd ..
```

3. Follow the installation instructions provided in the [VLN-CE-BERT](https://github.com/jacobkrantz/Sim2Sim-VLNCE/tree/main) repository to download the necessary data. **You only need to do steps 3–6 and download the model files from the "Model Downloads" section**.

&nbsp;

---

## Running Our Input-Adaptive Inference Method

To run our input-adaptive methods, execute the following command:

```bash
scripts/eval_ours.sh
```

Results will be saved in `data/results/RecVLNBERT-ce_vision`. To vary the configuration, refer to `sim2sim_vlnce/config/sgm-local_policy_efficient.yaml`. You can change the LSH similarity threshold by modifying `LSH_THRESHOLD` (and toggle it off by setting it to `1.0`).