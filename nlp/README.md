# Text Classification Training Pipeline

---

# Table of Contents
1. [Overview](#overview)  
2. [Datasets & Models](#datasets--models)
3. [Project Structure](#project-structure)  
4. [Setup & Usage](#setup--usage)  
5. [Training Configuration](#training-configuration)   
6. [Results & Evaluation](#results--evaluation)

---

# Overview
This project implements a **text classification pipeline** from scratch using **HuggingFace Transformers** and **PyTorch**.  

The pipeline includes:
- Data preprocessing & loading  
- Model definition  
- Training & Test loop  

The model and tokenizer are loaded using the `transformers` library, while the rest — including dataset preprocessing, training, and evaluation — is **implemented entirely with PyTorch**.  

The main goal is to implement the entire workflow — from configuration management to dataset processing, model training, and evaluation — to deepen understanding of each stage.

---

# Datasets & Models
This project uses the following dataset and models:

- **Dataset**: [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)  
- **Models**:  
  - `bert-base-uncased`  
  - `answerdotai/ModernBERT-base`  

> All models are loaded via the **HuggingFace Transformers** library using **pretrained weights**.

The original dataset consists of predefined train and test splits.
**However, in this project, we merged the original splits and then re-split them into train, validation, and test sets with a ratio of 8:1:1.**
Additionally, class proportions were preserved to prevent imbalance, ensuring fairness across all subsets.

#### Class Distribution Info
| Split          | Total Samples |        Label 0 |        Label 1 |
| -------------- | ------------: | -------------: | -------------: |
| **Train**      |        40,500 | 20,308 (50.1%) | 20,192 (49.9%) |
| **Validation** |         5,000 |  2,475 (49.5%) |  2,525 (50.5%) |
| **Test**       |         4,500 |  2,217 (49.3%) |  2,283 (50.7%) |


---

# Project Structure
```
nlp/
├── configs/
├── scripts/
├── src/
└── main.py
```

### 1. configs
YAML-based configuration management powered by [Hydra](https://github.com/facebookresearch/hydra), organized by function (dataset, model, training):

```
configs/
├── data
│   └── default.yaml
├── default.yaml
├── model
│   └── default.yaml
└── train
    └── default.yaml
```
This modular structure allows you to easily switch between datasets, models, and training setups by overriding configs from the command line.

### 2. src
```
src/
├── data.py        # Dataset loading & preprocessing
├── model.py       # Model definition
└── utils.py       # Utility functions
```

---

# Setup & Usage
This project was developed and tested in a **Linux environment**.

---

### Quick Start
**Step 1. Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2. Set WandB API Key**  
Edit `configs/default.yaml`:
```yaml
wandb_key: your_wandb_api_key
```

**Step 3. Run all training jobs**
```bash
bash scripts/run_all.sh
```

---

# Training Configuration

#### Data
- Dataset split: `train : valid : test = 0.8 : 0.1 : 0.1`  
- Batch size: `32`  
- Max sequence length: `128` (of tokenizer)

#### Optimizer
- Adam  
    - Learning rate: `5e-5` (with StepLR)  
    - Betas: `(0.9, 0.999)`  
    - Weight decay: `0`  

#### Model
- `bert-base-uncased`, `answerdotai/ModernBERT-base`  
- Dropout rate: `0.1`  

#### Train
- Epochs: `5`

# Results & Evaluation

### Experiment 1: BERT vs ModernBERT

#### 1. Test Data Evaluation
The following table summarizes the final accuracy of both models on the held-out test set:

| Model      | Test Accuracy |
| ---------- | ------------: |
| BERT       |         0.898 |
| ModernBERT |         0.914 |

- ModernBERT outperforms BERT by 1.6 percentage points in test accuracy, reinforcing the trend observed in validation performance and highlighting its stronger ability to generalize to unseen data.

#### 2. Training Curves — BERT vs ModernBERT

</div>
<div align="center">
  <img width="1000" alt="BERT" src="https://github.com/user-attachments/assets/43297680-53c9-4632-b86c-797c70af1460" />
</div>
</div>
<div align="center">
  <img width="1000" alt="ModernBERT" src="https://github.com/user-attachments/assets/af372f12-b8f9-48fb-ad4b-96073048f80c" />
</div>

- BERT (red) and ModernBERT (blue) both show a rapid increase in training accuracy within the first few hundred steps, reaching above 90% early on.
- ModernBERT converges slightly faster and maintains marginally higher training accuracy throughout.
- Training loss curves show both models steadily reducing loss, with ModernBERT achieving a lower final training loss — indicating more confident predictions on the training set.

#### 3. Validation Curves — Overfitting Patterns
<div align="center">
  <img width="1000" alt="ModernBERT" src="https://github.com/user-attachments/assets/b35467f1-90b3-423f-a2b4-0bf2a3003ee9" />
</div>

- Validation Loss: Both models show low loss early in training, but BERT’s loss increases more slowly than ModernBERT’s after ~3k steps. ModernBERT starts with lower loss but experiences more fluctuation and eventual rise, hinting at earlier overfitting.
- Validation Accuracy: ModernBERT maintains consistently higher validation accuracy (~0.91 peak) compared to BERT (~0.89 peak). However, its accuracy curve is less stable, possibly due to overfitting sensitivity or variance in batch composition.

#### 4. Key Insights

- Generalization: Despite slightly higher variance, ModernBERT sustains better validation accuracy overall, suggesting stronger generalization for this dataset.
- Training Efficiency: ModernBERT reaches high accuracy faster, which could be beneficial for time-constrained training scenarios.
- Overfitting Watch: ModernBERT’s earlier loss increase signals the need for earlier stopping or stronger regularization compared to BERT.


### Experiment 2: ModernBERT with Different Batch Size (Gradient Accumulation)

We conducted experiments to evaluate how gradient accumulation (i.e., effective batch size scaling) impacts the performance of ModernBERT.  
The base batch size was set to 32, and `gradient_accumulation_steps = 2, 8, 32` were used to simulate effective batch sizes of 64, 256, and 1024 respectively.

---

#### 1. Fixed Learning Rate

<div align="center">
  <img width="500" alt="ModernBERT" src="https://github.com/user-attachments/assets/ad918844-33dc-49a2-8ead-b0be2acf6337" />
</div>

| Model      | Batch size | Valid Acc | Test Acc (Best) | Test Acc (Last) | Training time | Time to baseline (0.913) |
| ---------- | ---------- | ---------- | --------------- | --------------- | ------------- | ------------------------ |
| ModernBERT | 64         | 0.915      | **0.918**       | **0.918**           | 37m 19s       | 20m 50s                  |
| ModernBERT | 256        | 0.913      | 0.915           | 0.915           | 36m 09s       | 20m 26s                  |
| ModernBERT | 1024       | 0.917      | 0.916           | 0.909           | **36m 07s**       | **19m 52s**                 |

- **Batch 64**: Produces the strongest best-test accuracy (0.918) and maintains it even after 5 epochs.  
- **Batch 256**: Slightly lower but stable, with no significant overfitting degradation.  
- **Batch 1024**: Matches strong validation accuracy but suffers from more noticeable overfitting (Best = 0.916 → Last = 0.909).
- With the same learning rate across all settings, the **batch size of 64 achieved the best performance** (Test Acc = 0.918).  
- Batch size 256 remained stable but slightly lower, while batch size 1024 showed strong validation accuracy but overfitted more quickly, resulting in lower final accuracy.
  

#### 2. Scaled Learning Rate (√ Rule)
- In practice, simply increasing the batch size while keeping the learning rate fixed can lead to **under-utilization of the larger batch**.  
- To address this, we applied the √ Rule for scaling the learning rate:
  
  $lr_{new} = lr_{base} \times \sqrt{\frac{B_{new}}{B_{base}}}$

  Where B is the batch size.
- The √ Rule was chosen instead of linear scaling, since experiments showed that scaling the learning rate linearly with the batch size caused **significant instability in training**.  

| Model      | Batch size | learning rate | Valid Acc (Best) | Test Acc (Best) | Test Acc (Last) | Training time |
| ---------- | ---------- | ------------- | ---------------- | --------------- | --------------- | ------------- |
| ModernBERT | 64         | 5e-5          | 0.915            | 0.917           | **0.918**           | 37m 19s       |
| ModernBERT | 256        | 1e-4          | 0.907            | 0.913           | 0.910           | 36m 41s       |
| ModernBERT | 1024       | 2e-4          | 0.915            | **0.921**           | 0.914           | 37m 10s       |

- Results demonstrate that with √ scaling, the model using the largest batch size (1024) achieved the **highest peak test accuracy (0.921)**.  
- This implies that as the batch size increases, the learning rate also needs to be increased to maintain effective gradient updates. In other words, √ scaling allows the model to **take advantage of larger batches by improving optimization efficiency while preserving stability**.


#### 3. Batch Size Comparison
| Model      | Batch size | learning rate | Valid Acc (Best) | Test Acc (Best) | Test Acc (Last) |
| ---------- | ---------- | ------------- | ---------------- | --------------- | --------------- |
| ModernBERT | 256        | 5e-5          | 0.913            | 0.915           | 0.915           |
| ModernBERT | 256        | 1e-4          | 0.907            | 0.907           | 0.910           |
| ModernBERT | 1024       | 5e-5          | 0.917            | 0.916           | 0.909           |
| ModernBERT | 1024       | 2e-4          | 0.915            | **0.922**           | 0.914           |

- For batch size 256, using the base learning rate (5e-5) yielded stable and consistent results (Best = 0.915, Last = 0.915). However, doubling the rate to 1e-4 caused a noticeable drop in both validation and test accuracy, indicating that medium batch sizes are less tolerant to aggressive learning rate scaling.

- For batch size 1024, the smaller rate (5e-5) gave solid validation accuracy but suffered from overfitting in later epochs (Best = 0.916 → Last = 0.909). In contrast, applying √ scaling (2e-4) improved both peak performance (0.922) and final stability (0.914), making it the most effective setting for large batches.
