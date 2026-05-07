![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Transfer Learning & Fine-Tuning

## Overview

Today you'll directly compare three ways to attack a small image classification problem:

1. **From scratch** — train a CNN with random initialisation
2. **Feature extraction** — frozen pretrained ResNet backbone, new head trained on your data
3. **Fine-tuning** — pretrained ResNet backbone unfrozen at the last block, trained with discriminative learning rates

You'll work with the **Oxford Flowers-102** dataset (102 classes of flowers, ~10–40 images per class) — small enough that overfitting from scratch is dramatic, large enough that transfer learning's improvement is unmistakable. The patterns you internalise here are exactly what you'll apply on Friday's cat-detection assessment.

## Learning Goals

By the end of this lab you should be able to:

- Load a pretrained model from `torchvision.models` and replace its classification head.
- Freeze and selectively unfreeze layers using `requires_grad`.
- Apply **discriminative learning rates** to different parameter groups in a single optimiser.
- Quantify the improvement transfer learning provides over training from scratch on a small dataset.
- Recognise the BatchNorm-during-fine-tuning pitfall and apply the standard mitigation.

## Setup and Context

You'll work in a single Jupyter Notebook. The Flowers-102 dataset is available via `torchvision.datasets.Flowers102` and downloads automatically.

## Requirements

### Fork and clone

1. Fork this repository to your own GitHub account.
2. Clone the fork to your local machine.
3. Navigate into the project directory.

### Python environment

```bash
pip install numpy pandas matplotlib torch torchvision scikit-learn
```

A GPU is helpful but not required. Each training run takes 5–15 minutes on CPU and 1–3 minutes on a modest GPU.

## Getting Started

1. Create a notebook called **`m6-07-transfer-learning-fine-tuning.ipynb`**.
2. Standard imports:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
```

3. Use the standard ImageNet preprocessing (this is critical for pretrained models):

```python
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize(232),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])
val_tf = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])
```

4. Load the Flowers-102 train/val/test splits (the dataset has its own canonical splits) into DataLoaders with batch size 32.

## Tasks

### Task 1 — From-Scratch Baseline

1. Build a small CNN from scratch (similar to the one from Day 03's lab) appropriate for 224×224 inputs and 102 output classes. Aim for roughly 1–3M parameters.
2. Train for 15 epochs with `Adam(lr=1e-3)` and cosine annealing. Use the train and val transforms above.
3. Track and plot training and validation loss + accuracy.
4. Report the best validation accuracy and the final test accuracy.

This is your baseline. Don't expect it to do well — that's the point.

### Task 2 — Feature Extraction with ResNet18

1. Load a pretrained ResNet18:

```python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
```

2. Freeze the entire backbone:

```python
for p in model.parameters():
    p.requires_grad = False
```

3. Replace the final classifier with a new linear layer for 102 classes:

```python
model.fc = nn.Linear(model.fc.in_features, 102)
```

4. Verify only the new `fc` layer's parameters require gradients. Print the number of trainable parameters vs total.
5. Train for 15 epochs with `Adam(lr=1e-3)`, applying the optimiser only to parameters with `requires_grad=True`:

```python
trainable = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(trainable, lr=1e-3)
```

6. Plot curves and report best validation and test accuracy.

### Task 3 — Fine-Tuning the Last Block

Take Task 2's trained model and continue training, but now unfreeze `layer4` (the last residual block) of the ResNet.

1. Set `requires_grad=True` for all parameters in `model.layer4` and `model.fc`. Leave the rest frozen.
2. Use **discriminative learning rates** — `lr=1e-5` for `layer4`, `lr=1e-3` for `fc`:

```python
optimizer = optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(),    "lr": 1e-3},
])
```

3. Train for an additional 10 epochs.
4. Plot the curves and report best validation and test accuracy.

### Task 4 — BatchNorm Stabilisation Experiment

Repeat Task 3 with one change: **freeze the BatchNorm layers** in `layer4` so they use their pretrained running statistics instead of the small mini-batch statistics.

```python
def freeze_bn(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

model.layer4.apply(freeze_bn)
```

Train for the same 10 additional epochs and report the test accuracy.

In a markdown cell, compare Task 3 and Task 4 results. On a small dataset like Flowers-102, freezing BN often gives a small-to-moderate improvement.

### Task 5 — Comparison Summary

Fill in this table and write a 4–6 sentence summary:

| Approach | Trainable params | Best val acc | Test acc | Total training time |
|---|---|---|---|---|
| From scratch (Task 1) | … | … | … | … |
| Feature extraction (Task 2) | … | … | … | … |
| Fine-tune last block (Task 3) | … | … | … | … |
| Fine-tune + BN frozen (Task 4) | … | … | … | … |

Your summary should answer:

- How much did transfer learning improve over from-scratch?
- Did fine-tuning the last block help over feature extraction?
- Did freezing BatchNorm help when fine-tuning?
- Which configuration would you recommend for tomorrow's cat-detection assessment, and why?

## Submission

### What to submit

- `m6-07-transfer-learning-fine-tuning.ipynb` — completed notebook.

### Definition of done (checklist)

- [ ] From-scratch baseline trained with curves and final test accuracy.
- [ ] Feature-extraction model trained with curves; trainable param count printed.
- [ ] Fine-tuned model with discriminative learning rates trained.
- [ ] BN-frozen variant of fine-tuning trained and compared.
- [ ] Comparison table with at least 4 rows and a written summary.
- [ ] `Kernel → Restart & Run All` produces no errors.

### How to submit (Git workflow)

```bash
git add .
git commit -m "lab: complete transfer learning fine-tuning"
git push origin main
```

Then open a **Pull Request** on the original repository with your comparison table and recommendation for the assessment.
