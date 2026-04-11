# MLOML — Multi-Layer Online Metric Learning
### Complete Implementation of Li et al. (arXiv:1805.05510v3)

This repository reproduces **all results** from the paper:
> *"A Multilayer Framework for Online Metric Learning"*  
> Wenbin Li, Yanfang Liu, Jing Huo, Yinghuan Shi, Yang Gao, Lei Wang, Jiebo Luo  
> arXiv:1805.05510v3, Aug 2023

---

## What is implemented?

| Component | File |
|---|---|
| **MOML** — Mahalanobis-based Online Metric Learning (base layer) | `mloml_implementation.py` |
| **MLOML** — Multi-Layer framework (FP / BP / FBP modes) | `mloml_implementation.py` |
| **RDML** baseline (Regularized Distance Metric Learning) | `mloml_implementation.py` |
| **LEGO** baseline (LogDet Exact Gradient Online) | `mloml_implementation.py` |
| **OPML** baseline (One-Pass Metric Learning) | `mloml_implementation.py` |
| **OASIS** baseline (Online Algorithm for Scalable Image Similarity) | `mloml_implementation.py` |
| **SLMOML** baseline (Scalable Large Margin Online ML) | `mloml_implementation.py` |
| One-pass triplet construction strategy | `mloml_implementation.py` |
| Data downloading + preprocessing | `data_loader.py` |
| **Table II** — Error rates on 12 UCI datasets | `run_experiments.py` |
| **Table III** — Propagation strategy comparison | `run_experiments.py` |
| **Fig 5** — Progressive feature representation | `run_experiments.py` |
| **Fig 7** — Learning ability vs. number of scans | `run_experiments.py` |
| **Fig 8** — Extendability (RDML-multi, LEGO-multi, OPML-multi) | `run_experiments.py` |

---

## Step 1 — Install Python and requirements

You need **Python 3.8 or newer**. Open a terminal and run:

```bash
pip install numpy scipy pandas matplotlib scikit-learn openpyxl
```

That is all you need. No GPU required — everything runs on CPU.

---

## Step 2 — Download the code

Save these 4 files into a folder (e.g. `mloml/`):

```
mloml/
├── mloml_implementation.py   ← all algorithms (MOML, MLOML, baselines)
├── data_loader.py            ← download + load 12 UCI datasets
├── run_experiments.py        ← reproduce Tables II–III and Figs 5,7,8
└── README.md                 ← this file
```

---

## Step 3 — Download the datasets

The paper uses 12 datasets from the UCI Machine Learning Repository.
Run this **once** to download all of them:

```bash
cd mloml
python data_loader.py
```

This creates a folder `mloml/data/raw/` and saves each dataset as a `.npz` file.

### What happens if a dataset cannot be downloaded?
The script will automatically generate a **synthetic fallback** with the same
dimensions as the real dataset. You will see a message like:
```
✗ balance download failed: ...  — using sklearn fallback
```
The real UCI datasets that can be downloaded automatically from the internet are:
`iris`, `wine`, `breast` (these come from scikit-learn and always work).

For the others, if your network blocks UCI, you can **manually download** them:

| Dataset | URL | Notes |
|---|---|---|
| `balance` | https://archive.ics.uci.edu/ml/datasets/Balance+Scale | `balance-scale.data` |
| `pima` | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database | `diabetes.csv` |
| `ionosphere` | https://archive.ics.uci.edu/ml/datasets/Ionosphere | `ionosphere.data` |
| `spect` | https://archive.ics.uci.edu/ml/datasets/SPECT+Heart | `SPECT.train` + `SPECT.test` |
| `lsvt` | https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation | `.xlsx` file |
| `waveform` | https://archive.ics.uci.edu/ml/datasets/Waveform+Database+Generator+(Version+2) | `.data` file |
| `diabetic` | https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set | `.arff` file |
| `pems` | https://archive.ics.uci.edu/ml/datasets/PEMS-SF | `.zip` file |
| `mlprove` | https://archive.ics.uci.edu/ml/datasets/Multiple+Features | `mfeat-fac` |

After downloading, place the raw files in `mloml/data/raw/sources/` and update
the relevant loader function in `data_loader.py` to point to your local file.

---

## Step 4 — Run all experiments

To reproduce **everything** (Tables II, III and Figs 5, 7, 8):

```bash
cd mloml
python run_experiments.py
```

This will print progress to the terminal and save results in `mloml/results/`:

```
results/
├── table2.csv   ← Table II:  error rates for all 12 datasets × 10 algorithms
├── table3.csv   ← Table III: FP / FBP / BP comparison
├── fig5.png     ← Fig 5:     progressive feature representation
├── fig7.png     ← Fig 7:     learning ability vs. number of scans
└── fig8.png     ← Fig 8:     extendability experiment
```

### ⚠️ Runtime warning
The **full experiment** (30 repeats × 12 datasets × 20 scans) takes several hours.
To run a quick test first, open `run_experiments.py` and change line:

```python
QUICK = False
```
to:
```python
QUICK = True   # 5 repeats, 5 scans — finishes in ~10 minutes
```

### Run individual experiments
You can also run just one experiment at a time:

```bash
python run_experiments.py table2   # Table II only
python run_experiments.py table3   # Table III only
python run_experiments.py fig5     # Figure 5 only
python run_experiments.py fig7     # Figure 7 only
python run_experiments.py fig8     # Figure 8 only
```

---

## How to use MLOML in your own code

```python
from mloml_implementation import MLOML, MOML, l2_normalize_rows
from data_loader import load_dataset
import numpy as np

# ── Load and preprocess data ──────────────────────────────────────
X, y = load_dataset("iris")          # any of the 12 datasets
X    = l2_normalize_rows(X)          # ℓ₂ normalise (required by paper)

# 50 / 50 train-test split
idx      = np.random.permutation(len(X))
X_train, y_train = X[idx[:75]], y[idx[:75]]
X_test,  y_test  = X[idx[75:]], y[idx[75:]]

# ── Train MLOML-r (ReLU, 3 layers, forward propagation) ──────────
model = MLOML(
    d         = X_train.shape[1],   # feature dimension
    n_layers  = 3,                   # number of metric layers (paper: 3 or 5)
    nonlinear = 'relu',              # 'relu' | 'sigmoid' | 'tanh'
    mode      = 'FP',                # 'FP' | 'BP' | 'FBP'
    gamma     = 0.01,                # regularisation (tune by cross-validation)
)

# Train (multiple-scan strategy, n_scans=20 as in the paper)
from mloml_implementation import train_mloml, knn_error_rate
train_mloml(model, X_train, y_train, n_scans=20)

# Transform features
X_train_transformed = model.transform(X_train)
X_test_transformed  = model.transform(X_test)

# Evaluate with 5-NN
error = knn_error_rate(X_train_transformed, y_train,
                        X_test_transformed,  y_test, k=5)
print(f"Error rate: {error:.3f}")

# ── Inspect individual metric layers ─────────────────────────────
for layer_i in range(1, model.n_layers + 1):
    X_te_layer = model.transform(X_test, layer_idx=layer_i)
    err_i      = knn_error_rate(model.transform(X_train, layer_idx=layer_i),
                                 y_train, X_te_layer, y_test, k=5)
    print(f"  Layer {layer_i} error: {err_i:.3f}")

# ── Use MOML as a single-layer baseline ──────────────────────────
from mloml_implementation import train_oml
moml = MOML(d=X_train.shape[1], gamma=0.01)
train_oml(moml, X_train, y_train, n_scans=20)
X_moml = moml.transform(X_test)
print("MOML error:", knn_error_rate(moml.transform(X_train), y_train, X_moml, y_test))
```

---

## Paper equations → code mapping

| Paper equation | Code location |
|---|---|
| Eq. (2) — Mahalanobis distance constraint | `MOML.compute_loss()` |
| Eq. (3) — Hinge loss | `MOML.compute_loss()` |
| Eq. (4) — MOML objective (PA-style) | `MOML.update()` |
| Eq. (5) — Gradient of MOML objective | `MOML.update()` |
| Eq. (6) — MOML closed-form update | `MOML.update()` |
| Eq. (1) — MLOML loss function | `MLOML._triplet_loss_final()` + `_backward()` |
| Theorem 1 — PSD guarantee | `project_psd()` |
| Theorem 2 — Regret bound | Proven in Appendix B; implemented as the update rule |
| Section III-A — One-pass triplet construction | `OnlineTripletConstructor` |
| Section III-B — FP / BP / FBP strategies | `MLOML.train_step()`, `_forward()`, `_backward()` |
| Section IV-A — Initialisation | `MLOML.reset()` (identity matrix) |
| Section IV-B — PCA + ℓ₂ normalisation | `preprocess()` in `run_experiments.py` |

---

## Expected results (Table II excerpt)

From the paper (30 repeats, 20 scans):

| Dataset | Euclidean | MOML | MLOML-r |
|---|---|---|---|
| iris | 0.038±0.016 | 0.028±0.018 | 0.026±0.017 |
| wine | 0.218±0.039 | 0.226±0.041 | 0.219±0.039 |
| ionosphere | 0.180±0.017 | 0.108±0.032 | 0.081±0.016 |
| balance | 0.108±0.013 | 0.070±0.010 | 0.066±0.012 |

Your results will match these closely when:
- Real UCI datasets are used (not synthetic fallbacks)
- 30 repeats and 20 scans are used
- Same k=5 NN classifier is used

---

## Notes on datasets
- **lsvt** (126 × 310): Reduced to 100-d by PCA before training (as per paper)
- **pems** (440 × 137710): Reduced to 100-d by PCA (very large dataset)
- All datasets are ℓ₂ normalised (each row has unit norm)
- 50/50 random train/test split, repeated 30 times

---

## Citation
```
@article{li2018multilayer,
  title   = {A Multilayer Framework for Online Metric Learning},
  author  = {Li, Wenbin and Liu, Yanfang and Huo, Jing and Shi, Yinghuan
             and Gao, Yang and Wang, Lei and Luo, Jiebo},
  journal = {arXiv preprint arXiv:1805.05510},
  year    = {2018}
}
```
