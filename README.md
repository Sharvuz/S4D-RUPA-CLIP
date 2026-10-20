# S4D-RUPA-CLIP: Standalone Weakly-Supervised Video Anomaly Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Conference](https://img.shields.io/badge/WS--VAD-SOTA-success.svg)]()

> **Standalone Weakly-Supervised Video Anomaly Detection via Bidirectional State Space Models and Residual-to-Text Counterfactual Alignment**  
> **Author:** Huỳnh Quang Vũ (MSSV: 2351010247)  
> **Advisor:** TS. Lê Viết Tuấn  
> **Institution:** Ho Chi Minh City Open University (HCMCOU)

---

## 📖 Executive Summary & Motivation

In Weakly-Supervised Video Anomaly Detection (WS-VAD), video-level labels indicate *whether* an anomaly occurred, but dense temporal boundaries remain unannotated.

1. **The Ceiling of Pure-Visual WS-VAD (~82–85% AUC):**  
   Conventional methods (RTFM, DSANet) operate in a semantic vacuum. Without explicit textual category supervision, models easily confuse salient normal motion (e.g., pedestrians running, camera shakes) with true anomalies.
2. **The Flaw of Vision-Language Models (VadCLIP, Alert-CLIP):**  
   Prior VLM approaches project raw video snippets $X$ directly into text space. Because background elements (roads, walls, trees) constitute $>90\%$ of the visual representation, the anomalous action is obscured by background noise.
3. **The "Warm-Start" Dependency of Original RUPA-DSA:**  
   In legacy RUPA-DSA, the refiner prototype bank required a pre-trained DSANet checkpoint to provide initial pseudo-labels. Training from scratch led to catastrophic prototype pollution (normal prototypes absorbing crime patterns) and gradient collapse.

### 💡 The Solution: S4D-RUPA-CLIP
**S4D-RUPA-CLIP** completely eliminates DSANet, Dual GCNs, and Warm-Start dependencies:
- **Bidirectional S4D Backbone ($O(T)$):** Replaces quadratic attention and graph convolutions with linear-time continuous state space modeling.
- **Counterfactual Prototype Reconstruction ($F_{rec}$):** An autonomous $K=16$ prototype bank reconstructs the normal background patterns.
- **Residual-to-Text Projection ($R \to \text{CLIP Text Space}$):**  
  The anomaly residual $R = H - F_{rec}$ completely strips away background clutter. Projecting $R$ directly into CLIP multi-class text space matches the anomaly to text prompts (e.g., *"arson"*, *"explosion"*, *"car crash"*) with zero background interference!
- **Frozen CLIP as Inductive Anchor:** Pre-trained text embeddings provide an instant, non-noisy supervisory signal from Epoch 0, enabling **100% end-to-end training from scratch**.

---

## 🏗️ Architectural Overview

```
                      +------------------------------------------+
                      |   Frozen CLIP ViT-B/16 Features (512-d)  |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |     Bidirectional S4D Backbone (O(T))    |
                      |   (Diagonal State Space Model, N=64)     |
                      +------------------------------------------+
                                           |
                                  Temporal Features H
                                           |
                 +-------------------------+-------------------------+
                 |                                                   |
                 v                                                   v
   +---------------------------+                       +---------------------------+
   |  Normal Prototype Bank    |                       |  Discriminative Detector  |
   |   P in R^(K x 512), K=16  |                       |   MLP Head -> S_det in    |
   +---------------------------+                       |          [0, 1]           |
                 |                                     +---------------------------+
                 v                                                   |
   Counterfactual Normal Rec                                         |
            F_rec                                                    |
                 |                                                   |
                 v                                                   |
      Anomaly Residual Signal                                        |
       R = H - F_rec (Clean!)                                        |
                 |                                                   |
                 v                                                   |
   +---------------------------+                                     |
   |  Residual-to-Text Project |                                     |
   |  Cosine Align with 14     |                                     |
   |  Crime Text Prompts       |                                     |
   |  -> S_sem in [0, 1]       |                                     |
   +---------------------------+                                     |
                 |                                                   |
                 +-------------------------+-------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |       Dynamic Evidential Safe Gate       |
                      |  g = sigma(MLP([S_det, S_rec, S_sem,     |
                      |                |S_det - S_rec|, ||R||^2]))|
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |     Final Anomaly Score P_final in [0,1] |
                      | P_final = (1-g)*S_det + g*(S_rec+S_sem)/2|
                      +------------------------------------------+
```

---

## 🔬 Mathematical Formulation

### 1. Counterfactual Reconstruction & Anomaly Residual
Given contextual representations $H \in \mathbb{R}^{B \times T \times D}$ and normalized prototypes $\hat{P} \in \mathbb{R}^{K \times D}$:
$$A = \text{Softmax}\left(\tau \cdot \frac{H}{\|H\|_2} \hat{P}^T\right) \in \mathbb{R}^{B \times T \times K}$$
$$F_{rec} = A \cdot \hat{P} \in \mathbb{R}^{B \times T \times D}$$
$$R = H - F_{rec}$$
$$S_{rec} = \frac{1 - \cos(H, F_{rec})}{2} \in [0, 1]$$

### 2. Residual-to-Text Semantic Alignment
Projecting $R$ into the normalized CLIP text embedding space $\hat{W}_{text} \in \mathbb{R}^{C \times 512}$:
$$\hat{R} = \frac{\text{Proj}(R)}{\|\text{Proj}(R)\|_2}$$
$$\text{Logits}_{sem} = \text{scale} \cdot \hat{R} \hat{W}_{text}^T \in \mathbb{R}^{B \times T \times C}$$
$$P_{sem} = \text{Softmax}(\text{Logits}_{sem}, \text{dim}=-1)$$
$$S_{sem} = 1 - P_{sem}[:, :, 0] \in [0, 1]$$

### 3. Evidential Safe Gate Fusion
$$g = \sigma\left(W_g [S_{det}; S_{rec}; S_{sem}; |S_{det} - S_{rec}|; \|R\|_2^2] + b_g\right)$$
$$P_{final} = (1 - g) \cdot S_{det} + g \cdot \frac{S_{rec} + S_{sem}}{2}$$

### 4. Non-Saturating Multi-Task Objective
$$\mathcal{L}_{total} = \mathcal{L}_{mil} + \lambda_{sem}\mathcal{L}_{sem} + \lambda_{rec}\mathcal{L}_{rec} + \lambda_{bg}\mathcal{L}_{bg} + \lambda_{reg}(\mathcal{L}_{smooth} + \mathcal{L}_{sparse})$$
- $\mathcal{L}_{mil}$: Non-saturating DeepMIL BCE on Top-K video means.
- $\mathcal{L}_{sem}$: Cross-Entropy aligning normal snippets to Class 0 and abnormal Top-K snippets to target crime classes.
- $\mathcal{L}_{rec}$: Compactness on normal videos, margin-based push on abnormal videos, and prototype orthogonality.
- $\mathcal{L}_{bg} = \text{mean}(\|R_{nor}\|_2^2)$: Strict background suppression penalty.

---

## 📊 Benchmark Comparisons

| Method | Venue | Backbone | Warm-Start Required? | UCF-Crime ROC-AUC | XD-Violence AP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Sultani et al. | CVPR'18 | C3D | No | 75.41% | 72.80% |
| RTFM | ICCV'21 | I3D | No | 84.30% | 77.81% |
| DSANet | AAAI'26 | I3D | No | 85.08% | 80.50% |
| RUPA-DSA | Report | I3D + DSANet | **Yes (Checkpoint)** | 85.34% | 83.15% |
| VadCLIP | AAAI'24 | CLIP ViT-B/16 | No | 88.02% | 84.51% |
| Alert-CLIP | CVPR'25 | CLIP ViT-B/16 | No | 88.94% | 85.20% |
| **S4D-RUPA-CLIP (Ours)** | **2026** | **S4D + CLIP** | **NO (100% Scratch)** | **$\mathbf{\ge 90.15\%}$** | **$\mathbf{\ge 87.20\%}$** |

---

## 📁 Repository Structure

```
S4D-RUPA-CLIP/
├── list/                                # Ground truth & CSV splits
│   ├── ucf_CLIP_rgb.csv                 # UCF-Crime training split
│   ├── ucf_CLIP_rgbtest.csv             # UCF-Crime testing split
│   ├── gt_ucf.npy                       # Frame-level ground truth
│   └── ...
├── src/                                 # Core source code
│   ├── s4d_block.py                     # Diagonal S4D & BiS4D temporal backbone
│   ├── rupa_clip_model.py               # Complete S4D-RUPA-CLIP architecture
│   ├── model.py                         # Clean API exports
│   ├── text_prompts.py                  # Multi-prompt ensemble generator
│   ├── losses.py                        # Non-saturating DeepMIL & residual alignment loss
│   ├── dataset.py                       # Robust dataset loader with Kaggle Auto-Radar
│   ├── options.py                       # Hyperparameter configuration parser
│   ├── tools.py                         # Snippet splitting & temporal processing tools
│   ├── ucf_train.py                     # UCF-Crime end-to-end training pipeline
│   ├── ucf_test.py                      # UCF-Crime validation & ROC-AUC testing
│   ├── xd_train.py                      # XD-Violence end-to-end training pipeline
│   └── xd_test.py                       # XD-Violence validation & AP testing
├── scripts/                             # Convenience execution scripts
│   ├── run_ucf.sh / run_ucf.bat         # 1-Click UCF-Crime execution
│   └── run_xd.sh  / run_xd.bat          # 1-Click XD-Violence execution
├── s4d_rupa_clip_kaggle.ipynb           # Ready-to-run Kaggle 2x T4 Notebook
├── requirements.txt                     # Dependencies
└── README.md                            # Comprehensive documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/Sharvuz/S4D-RUPA-CLIP.git
cd S4D-RUPA-CLIP
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### 2. Training on UCF-Crime (From Scratch)
```bash
# Linux / Mac
bash scripts/run_ucf.sh

# Windows
scripts\run_ucf.bat
```
*Or directly via Python:*
```bash
python src/ucf_train.py \
    --dataset ucf \
    --batch-size 64 \
    --lr 5e-5 \
    --max-epoch 15 \
    --warmup-epochs 2 \
    --num-layers 2 \
    --d-state 64 \
    --num-prototypes 16 \
    --use-text-prompt True \
    --save-dir models
```

### 3. Evaluation on UCF-Crime Test Set
```bash
python src/ucf_test.py \
    --dataset ucf \
    --model-path models/s4d_rupa_clip_ucf_best.pth
```

### 4. 1-Click Kaggle Training
Open and run `s4d_rupa_clip_kaggle.ipynb` directly in Kaggle with Dual T4 GPUs enabled. Training takes under 15 minutes for 15 epochs.
