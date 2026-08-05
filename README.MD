

# Suppress and Diversify: Refining Robust Pathways for Corruption Robustness (ICML 2026 Spotlight)

🎯 **Official PyTorch Implementation of the ICML 2026 Spotlight Paper.**

---

## 📢 News
* **[2026/07]** 🎉 Our paper has been accepted as a **Spotlight** at ICML 2026! 
* **[2026/07]** 🚀 Core training pipeline, configurations, and environment dependencies are now released. Full pre-trained checkpoints and logs will be progressively integrated. ⭐ **Star this repository to stay tuned!**

---

## 🛠️ Environment Setup

We recommend using [Anaconda](https://www.anaconda.com/) to build a clean environment. Run the following commands sequentially to set up the dependencies:

```bash
# 1. Create and activate a clean virtual environment
conda create -n sd python=3.10 -y
conda activate sd

# 2. Install PyTorch and Torchvision with CUDA 11.8 support
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 3. Install the remaining dependencies
pip install -r requirements.txt

```

---

## 📦 Pre-trained Model Checkpoints

We provide the pre-trained ResNet-50 weights corresponding to different methods evaluated in our paper. You can download them from the links below and place them in your local directory (e.g., `checkpoints/`):

| Model Architecture | Method | Download Link |
| :--- | :--- | :--- |
| ResNet-50 | SD | [Click to Download (Extraction Code: icml2026)](https://pan.baidu.com/s/13ynkzFKC7myL_gtMLY9K3w) |
| ResNet-50 | AugMix + SD | [Click to Download (Extraction Code: icml2026)](https://pan.baidu.com/s/13ynkzFKC7myL_gtMLY9K3w) |

---

## 🚀 Training and Evaluation Pipelines

All experiments are managed via YAML configurations located in the `cfgs/` directory.

### 1. Training from Scratch

We provide a unified bash script `script.sh` to kick off multi-GPU training via `torchrun`. You can uncomment the corresponding lines in `script.sh` to switch between different training schemes:

```bash
# Execute the training script
bash script.sh

```

*Inside `script.sh`, the configuration routing is defined as:*

* `resnet50`: Standard Baseline training setup (`cfgs/resnet50_baseline.yaml`)
* `resnet50_sd`: Suppress and Diversify training (`cfgs/resnet50_sd.yaml`)
* `resnet50_augmix`: AugMix training setup (`cfgs/resnet50_augmix.yaml`)
* `resnet50_augmix_sd`: Combined AugMix and SD training (`cfgs/resnet50_augmix+sd.yaml`)

### 2. Evaluation / Inference

To evaluate a pre-trained checkpoint or run test-time inference (e.g., testing on corruption datasets like ImageNet-C):

1. Open the corresponding test configuration file in `cfgs/` (e.g., `cfgs/resnet50_sd_test.yaml` or `cfgs/resnet50_augmix+sd_test.yaml`).
2. Locate the model_path field and fill in the local path to your downloaded .pth checkpoint. Also, set test_only_ood to True in the YAML configuration file.
```yaml
# Example snippet inside cfgs/resnet50_sd_test.yaml
model_path: "path/to/your/downloaded/resnet50_sd.pth"
test_only_ood: True
```



