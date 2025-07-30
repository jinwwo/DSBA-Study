# Image Classification Training Pipeline

---

# 📚 Table of Contents

1. [Overview](#-overview)
2. [Datasets & Models](-#-datasets--models)  
3. [Project Structure](-#-project-structure)  
4. [Setup & Usage](-#-setup--usage)  
5. [Training Configuration](#-training-configuration)  
6. [Results & Evaluation](#-results--evaluation)

---

# 📌 Overview

This project implements a **modular image classification pipeline** from scratch, covering:

- Data preprocessing & loading  
- Model definition  
- Training loop  
- Evaluation with distribution-shifted test data

All components were designed and integrated independently, without relying on pretrained weights or high-level training frameworks. The aim was to deepen understanding of pipeline orchestration, and to allow flexible extensibility across datasets and model types.

---

# 🗂️ Datasets & Models

Three datasets were used throughout this project:

- **CIFAR-10**: Main dataset used for training and evaluation of all models.
- [**CIFAR-10.1**](https://github.com/modestyachts/CIFAR-10.1): Used exclusively for evaluating robustness under **distribution shift**, providing insight into generalization beyond the original training data.
- **Tiny ImageNet**: Integrated to test scalability and modular generalization to more complex, hierarchical datasets, though not used for training in the current scope.

> All models were trained **from scratch**, without pretrained weights.

Models were grouped into **CNN-based** and **Transformer-based** architectures, and organized by publication year for clarity:
#### CNN-based Models:
| Model         | Year | Reason for Inclusion |
|---------------|------|----------------------|
| ResNet-18     | 2015 | Classic baseline, low-depth ResNet for efficient training |
| ResNet-50     | 2015 | Deeper ResNet to compare scaling behavior |
| DenseNet      | 2017 | Parameter-efficient architecture with feature reuse |
| MobileNet     | 2017 | Lightweight model optimized for mobile/low-power settings |
| EfficientNet-B0 | 2019 | Modern compound-scaled CNN, good tradeoff between accuracy and efficiency |

#### Transformer-based Models:
| Model         | Year | Reason for Inclusion |
|---------------|------|----------------------|
| ViT-Tiny      | 2020 | Vision Transformer baseline, minimal size for experimentation |
| ViT-Small     | 2020 | Slightly larger ViT for scaling analysis |
| DeiT-Tiny     | 2021 | Data-efficient ViT trained without large-scale pretraining |

> The selection aims to cover a variety of architectural paradigms (depth, width, convolutional vs attention-based) and efficiency/accuracy tradeoffs.

---

# 📁 Project Structure

```
workspace/
├── configs/
├── src/
├── main.py
└── run_all.sh
```

### 1. configs

The project uses **YAML-based configuration management powered by [Hydra](https://github.com/facebookresearch/hydra)**, structured by function (dataset, model, training):

```
configs/
├── default.yaml
├── data/
│   ├── dataset/
│   │   ├── cifar10.yaml
│   │   └── ...
│   └── default.yaml
├── model/
│   ├── default.yaml
│   └── models/
│       ├── resnet18.yaml
│       └── ...
└── train/
    ├── default.yaml
    ├── logger/
    │   └── default.yaml
    ├── loss/
    │   ├── cross_entropy.yaml
    │   └── ...
    ├── optimizer/
    │   ├── adam.yaml
    │   └── ...
    └── scheduler/
        ├── cosine.yaml
        └── ...
```

This modular configuration allows you to easily:
- Switch between models or datasets
- Try different optimizers/schedulers
- Tune hyperparameters consistently

Hydra enables dynamic overriding of any configuration element from the command line or programmatically, which allows for scalable experimentation workflows.

### 2. src

```
src
├── dataset/
│   ├── dataset.py                # Dataset loading interface
│   ├── loader.py                 # Unified DataLoader factory
│
├── model/
│   ├── builder.py                # Model builder
│   └── model_zoo.py              # Implementation or import wrapper for all models
│
├── trainer/
│   ├── callbacks/
│   │   ├── base.py
│   │   ├── early_stopping.py
│   │   ├── lr_monitor.py
│   │   └── model_checkpoint.py
│   └── trainer.py               # Main training loop abstraction
│
└── utils/
    ├── factory.py               # Generic factory pattern for config-object binding
    ├── metric.py                # Accuracy, top-k, loss tracking
    └── utils.py                 # Seed setup, logging, etc.
```

# ⚙️ Setup & Usage

This project was developed and tested in a **Linux environment**.

> **Optional: Check your system info**
>
> ```bash
> cat /etc/os-release
> uname -r
> lscpu
> nvidia-smi
> ```
> Example Output:
> ```
> OS: Ubuntu 20.04.6 LTS
> Kernel: 5.15.0-67-generic
> GPU: NVIDIA GeForce RTX 2080 Ti
> ```

---

### 🚀 Quick Start

**Step 1. Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2. Prepare datasets**
Place your dataset files in the following path:
```bash
/workspace/dataset
```

**Step 3. Set WandB API key**
Edit `configs/default.yaml`:
```yaml
wandb_key: your_wandb_api_key
```

**Step 4. Run all training jobs**
```bash
bash run_all.sh
```

---

### Output

All results will be saved to:

```
/workspace/checkpoints/
```

Saved files include:

| File                        | Description                         |
|-----------------------------|-------------------------------------|
| `best_model.pt`            | Best model (based on validation)   |
| `last_model.pt`            | Model after final epoch            |
| `best_results.json`        | Evaluation on validation set       |
| `test_results.json`        | Final test accuracy                |
| `test_cifar10.1.json`      | Evaluation on CIFAR-10.1           |
| `test_results-per_class.json` | Per-class test metrics          |
| `configs.yaml`             | Snapshot of config used            |
| `history.json`             | Training history (loss/acc)        |


# 📋 Training Configuration

The default training configuration used throughout this project is defined as follows via `configs/default.yaml` and related model-specific YAML files:

#### Data
- Datasets: CIFAR-10, CIFAR-10.1  
- Input Size: 224×224  
- Epochs: 50  
- Batch Size: 64  
- Loss Function: Cross Entropy Loss  

#### Optimizer
- Adam  
    - Learning Rate: 0.001 (with CosineAnnealingLR)  
    - Betas: (0.9, 0.999)  
    - Eps: 1e-8  
    - Weight Decay: 1e-4  

#### Models Used

- CNN-based: `resnet18`, `resnet50`, `densenet121`, `mobilenetv3_small_050`, `efficientnet_b0`  
- ViT-based: `vit_tiny_patch16_224`, `vit_base_patch16_224`, `deit_tiny_patch16_224`  

#### Evaluation Metrics

- Accuracy (Top-1)  
- F1-score  
- AUROC  
- Precision

# 📊 Results & Evaluation

### Experiment 1. Comparison of Overfitting Patterns between CNNs and ViTs

Evaluating model performance based solely on final accuracy or loss is often insufficient.
To better understand how well a model generalizes and how sensitive it is to overfitting, we need to examine the trends of both training and validation loss/accuracy over time.

CNNs and Vision Transformers (ViTs) differ significantly in their architectural inductive biases, which can lead to different training dynamics and overfitting behaviors.
This experiment compares the training curves of various models to analyze:
- how long each model maintains generalization,
- how early and severely overfitting occurs,
- and whether early stopping thresholds can be inferred based on loss divergence.

| model                  |  turning point |  val_loss  |  val_acc  |  val_acc*  | val_acc*@epoch | test_loss | test_acc |
|------------------------|:----:|:----------:|:---------:|:----------:|:---------------:|:---------:|:--------:|
| resnet18               |  18  |  0.517226  |  0.8801   |  0.8801    |       49        | 0.552384  |  0.8726  |
| resnet50               |  -1  |  0.544705  |  0.8631   |  0.8634    |       46        | 0.530928  |  0.8640  |
| densenet121            |  19  |  0.359095  |  0.9173   |  0.9186    |       47        | 0.364581  |  0.9186  |
| mobilenetv3_small_050  |  20  |  0.909682  |  0.8308   |  0.8386    |       48        | 0.935850  |  0.8254  |
| efficientnet_b0        |  19  |  0.417452  |  0.9001   |  0.9001    |       49        | 0.454946  |  0.8959  |
| vit_tiny_patch16_224   |  27  |  1.313299  |  0.6708   |  0.6790    |       35        | 1.386092  |  0.6631  |
| deit_tiny_patch16_224  |  30  |  1.189161  |  0.6777   |  0.6814    |       45        | 1.217820  |  0.6662  |

<div align="center">
  <img width="1000" alt="exp1" src="https://github.com/user-attachments/assets/6c7b0560-04ff-4045-9f65-0df812534ac7" />
</div>

---

### Experiment 2. Comparison of Feature Representations

Deep learning models learn high-dimensional feature representations from input images.
These features are used for final classification and provide insight into how the model understands and separates different classes in the dataset.

In this experiment, we visually analyze how CNNs and ViTs construct their feature spaces using the same dataset.
We focus on how well different classes are separated in the learned space, revealing differences in representation strategies between architectures.

</div>

<div align="center">
  <img width="1000" alt="exp2" src="https://github.com/user-attachments/assets/cfd75ea3-af4b-4113-9af0-4566aa496e6a" />
</div>

---

### Experiment 3. Generalization under Domain Shift

While deep learning models are typically evaluated on test sets drawn from the same distribution as the training data, real-world applications often involve domain shift — caused by changes in lighting, style, equipment, time, etc.

This experiment evaluates how models trained on CIFAR-10 generalize to the shifted CIFAR-10.1 dataset.
We also explore whether applying simple Test-Time Adaptation (TTA) techniques can help recover performance under domain shift conditions.

<div align="center">
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>CIFAR-10 Acc</th>
      <th>CIFAR-10.1 Acc</th>
      <th>Δ Accuracy (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center">vit_tiny_patch16_224</td><td align="center">0.6631</td><td align="center">0.5275</td><td align="center">13.56</td></tr>
    <tr><td align="center">deit_tiny_patch16_224</td><td align="center">0.6662</td><td align="center">0.5430</td><td align="center">12.32</td></tr>
    <tr><td align="center">resnet50</td><td align="center">0.8640</td><td align="center">0.7535</td><td align="center">11.05</td></tr>
    <tr><td align="center">resnet18</td><td align="center">0.8726</td><td align="center">0.7700</td><td align="center">10.26</td></tr>
    <tr><td align="center">mobilenetv3_small_050</td><td align="center">0.8254</td><td align="center">0.7240</td><td align="center">10.14</td></tr>
    <tr><td align="center">efficientnet_b0</td><td align="center">0.8959</td><td align="center">0.8015</td><td align="center">9.44</td></tr>
    <tr><td align="center">densenet121</td><td align="center">0.9186</td><td align="center">0.8280</td><td align="center">9.06</td></tr>
  </tbody>
</table>

<div align="center">
  
  <img width="1000" src="https://github.com/user-attachments/assets/3631b64a-e54a-4814-9285-ca95208f8e5b" />
</div>
