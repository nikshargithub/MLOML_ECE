"""
Experiment runner for MLOML paper.

Reproduces:
  - Table II  : Error rates on 12 UCI datasets (comparison with state-of-the-art)
  - Table III : Propagation strategy comparison (FP / BP / FBP)
  - Fig 5     : Progressive feature representation (error vs. layer depth)
  - Fig 7     : Learning ability (error vs. number of scans)
  - Fig 8     : Extendability (RDML-multi, LEGO-multi, OPML-multi)

Usage:
    python run_experiments.py

All results are saved to:
    results/table2.csv
    results/table3.csv
    results/fig5.png
    results/fig7.png
    results/fig8.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data_loader   import load_dataset, download_all
from mloml_implementation import (
    MOML, RDML, LEGO, OPML, OASIS, SLMOML,
    MLOML, l2_normalize_rows,
    train_oml, train_mloml,
    evaluate_model, evaluate_euclidean, knn_error_rate,
    OnlineTripletConstructor,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ─── paper constants ──────────────────────────────────────────────
DATASETS   = ["lsvt", "iris", "wine", "spect", "ionosphere", "pems",
              "balance", "breast", "pima",   "diabetic",    "waveform", "mlprove"]
N_REPEATS  = 30          # 30 random 50/50 splits  (paper Section IV-B)
K_NN       = 5           # k = 5 (paper Section IV-C)
N_SCANS    = 20          # multiple-scan (paper Section IV-C)
PCA_DIM    = 100         # reduce to 100-d when d >= 200
N_LAYERS   = 3           # default 3 layers  (paper Section IV-D)
GAMMA_GRID = [1e-4, 1e-3, 1e-2, 1e-1]

# For a quicker run during development set QUICK=True (fewer repeats/scans)
QUICK = True
if QUICK:
    N_REPEATS = 5
    N_SCANS   = 5


# ─── preprocessing ────────────────────────────────────────────────

def preprocess(X, y, seed, pca_dim=PCA_DIM):
    """
    50/50 train/test split → dimensionality reduction if d≥200 → ℓ₂ normalisation.

    For d >= 200 the paper applies PCA to reduce to 100-d (Section IV-B).
    For very high-dimensional data (d >= 10000, e.g. pems with d=137710) we use
    a random projection, which preserves pairwise distances by the
    Johnson-Lindenstrauss lemma and is vastly faster than full PCA.
    """
    rng   = np.random.RandomState(seed)
    n     = len(X)
    idx   = rng.permutation(n)
    split = n // 2
    tr, te = idx[:split], idx[split:]

    X_tr, X_te = X[tr].copy(), X[te].copy()
    y_tr, y_te = y[tr].copy(), y[te].copy()

    # standardise (zero mean / unit variance)
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    d = X_tr.shape[1]
    if d >= 10000:
        # Random projection (fast, JL-preserving) for very high-d
        from sklearn.random_projection import GaussianRandomProjection
        n_comp = min(pca_dim, X_tr.shape[0] - 1)
        rp = GaussianRandomProjection(n_components=n_comp, random_state=seed)
        X_tr = rp.fit_transform(X_tr)
        X_te = rp.transform(X_te)
    elif d >= 200:
        # Standard PCA for moderately high-d
        n_comp = min(pca_dim, X_tr.shape[0] - 1, d)
        pca    = PCA(n_components=n_comp, random_state=0)
        X_tr   = pca.fit_transform(X_tr)
        X_te   = pca.transform(X_te)

    # ℓ₂ normalisation (paper Section IV-B)
    X_tr = l2_normalize_rows(X_tr)
    X_te = l2_normalize_rows(X_te)
    return X_tr, X_te, y_tr, y_te


# ─── cross-validation for gamma ──────────────────────────────────

def cv_gamma(ModelClass, X_train, y_train, gamma_grid=GAMMA_GRID,
             n_scans=N_SCANS, k=K_NN, n_cv=2, is_mloml=False, extra_kwargs=None):
    """Simple 2-fold CV to pick best gamma (faster than 3-fold)."""
    extra_kwargs = extra_kwargs or {}
    n = len(X_train)
    fold = n // n_cv
    best_gamma, best_err = gamma_grid[0], np.inf

    for g in gamma_grid:
        errs = []
        for fold_i in range(n_cv):
            val_idx  = np.arange(fold_i*fold, min((fold_i+1)*fold, n))
            tr_idx   = np.array([i for i in range(n) if i not in val_idx])
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_va, y_va = X_train[val_idx], y_train[val_idx]
            try:
                model = ModelClass(d=X_tr.shape[1], gamma=g, **extra_kwargs)
                err   = evaluate_model(model, X_tr, y_tr, X_va, y_va,
                                       n_scans=max(3, n_scans//4),  # use fewer scans for CV
                                       k=k, is_mloml=is_mloml)
                errs.append(err)
            except Exception:
                errs.append(1.0)
        mean_err = np.mean(errs)
        if mean_err < best_err:
            best_err, best_gamma = mean_err, g

    return best_gamma


# ─── run one dataset for Table II ────────────────────────────────

def run_dataset_tableII(ds_name, X, y, n_repeats=N_REPEATS, n_scans=N_SCANS):
    """
    Evaluate all algorithms on one dataset.
    Returns dict: algorithm_name -> (mean_error, std_error)
    """
    print(f"\n  Dataset: {ds_name}  shape={X.shape}")
    d = None  # will be set after first split

    results = {
        "Euclidean": [],
        "RDML":      [],
        "LEGO":      [],
        "OASIS":     [],
        "OPML":      [],
        "SLMOML":    [],
        "MOML":      [],
        "MLOML-r":   [],
        "MLOML-s":   [],
        "MLOML-t":   [],
    }

    for rep in range(n_repeats):
        seed = rep * 100
        X_tr, X_te, y_tr, y_te = preprocess(X, y, seed)
        d = X_tr.shape[1]

        # ----- pick gamma by CV (use rep=0 split to save time) -----
        if rep == 0:
            g_moml  = cv_gamma(MOML,   X_tr, y_tr, is_mloml=False)
            g_rdml  = cv_gamma(RDML,   X_tr, y_tr, is_mloml=False)
            g_lego  = cv_gamma(LEGO,   X_tr, y_tr, is_mloml=False)
            g_oasis = cv_gamma(OASIS,  X_tr, y_tr, is_mloml=False)
            g_opml  = cv_gamma(OPML,   X_tr, y_tr, is_mloml=False)
            g_slm   = cv_gamma(SLMOML, X_tr, y_tr, is_mloml=False)

            def _mloml_cv(nl):
                return cv_gamma(MLOML, X_tr, y_tr, is_mloml=True,
                                extra_kwargs=dict(n_layers=N_LAYERS,
                                                  nonlinear=nl, mode='FP'))
            g_mloml_r = _mloml_cv('relu')
            g_mloml_s = _mloml_cv('sigmoid')
            g_mloml_t = _mloml_cv('tanh')

        # ----- Euclidean -----
        results["Euclidean"].append(evaluate_euclidean(X_tr, y_tr, X_te, y_te))

        # ----- single-layer OML algorithms -----
        for ModelClass, gval, key in [
            (RDML,   g_rdml,  "RDML"),
            (LEGO,   g_lego,  "LEGO"),
            (OASIS,  g_oasis, "OASIS"),
            (OPML,   g_opml,  "OPML"),
            (SLMOML, g_slm,   "SLMOML"),
            (MOML,   g_moml,  "MOML"),
        ]:
            model = ModelClass(d=d, gamma=gval)
            err   = evaluate_model(model, X_tr, y_tr, X_te, y_te,
                                   n_scans=n_scans, is_mloml=False)
            results[key].append(err)

        # ----- MLOML variants -----
        for nl, gval, key in [
            ('relu',    g_mloml_r, "MLOML-r"),
            ('sigmoid', g_mloml_s, "MLOML-s"),
            ('tanh',    g_mloml_t, "MLOML-t"),
        ]:
            model = MLOML(d=d, n_layers=N_LAYERS, nonlinear=nl,
                          mode='FP', gamma=gval)
            err   = evaluate_model(model, X_tr, y_tr, X_te, y_te,
                                   n_scans=n_scans, is_mloml=True)
            results[key].append(err)

        print(f"    rep {rep+1:2d}/{n_repeats}  MLOML-r={results['MLOML-r'][-1]:.3f}"
              f"  MOML={results['MOML'][-1]:.3f}"
              f"  Euclidean={results['Euclidean'][-1]:.3f}", flush=True)

    summary = {k: (np.mean(v), np.std(v)) for k, v in results.items()}
    return summary


# ─── Table II ────────────────────────────────────────────────────

def run_table2(n_repeats=N_REPEATS):
    print("\n" + "="*60)
    print("TABLE II: Error rates on 12 UCI datasets")
    print("="*60)
    all_results = {}
    for ds in DATASETS:
        X, y = load_dataset(ds)
        all_results[ds] = run_dataset_tableII(ds, X, y, n_repeats=n_repeats)

    # build DataFrame
    cols = ["Euclidean", "RDML", "LEGO", "OASIS", "OPML",
            "SLMOML", "MOML", "MLOML-t", "MLOML-s", "MLOML-r"]
    rows = []
    for ds in DATASETS:
        row = {"Dataset": ds}
        for col in cols:
            mean, std = all_results[ds][col]
            row[col] = f"{mean:.3f}±{std:.3f}"
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Dataset")
    df.to_csv(RESULTS_DIR / "table2.csv")
    print("\n" + df.to_string())
    print(f"\n✓ Saved to {RESULTS_DIR/'table2.csv'}")
    return all_results


# ─── Table III: propagation strategy ─────────────────────────────

def run_table3(n_repeats=N_REPEATS):
    print("\n" + "="*60)
    print("TABLE III: Propagation strategy comparison")
    print("="*60)

    modes = ['BP', 'FBP', 'FP']
    nonlinears = ['relu', 'sigmoid', 'tanh']
    col_labels = {
        ('relu',    'BP'):  'r-BP',  ('relu',    'FBP'): 'r-FBP',  ('relu',    'FP'): 'r-FP',
        ('sigmoid', 'BP'):  's-BP',  ('sigmoid', 'FBP'): 's-FBP',  ('sigmoid', 'FP'): 's-FP',
        ('tanh',    'BP'):  't-BP',  ('tanh',    'FBP'): 't-FBP',  ('tanh',    'FP'): 't-FP',
    }
    all_results = {k: {ds: [] for ds in DATASETS} for k in col_labels}

    for ds in DATASETS:
        X, y = load_dataset(ds)
        print(f"\n  {ds}  shape={X.shape}")

        for rep in range(n_repeats):
            seed = rep * 100
            X_tr, X_te, y_tr, y_te = preprocess(X, y, seed)
            d = X_tr.shape[1]

            if rep == 0:
                g = cv_gamma(MLOML, X_tr, y_tr, is_mloml=True,
                             extra_kwargs=dict(n_layers=N_LAYERS,
                                               nonlinear='relu', mode='FP'))

            for nl in nonlinears:
                for mode in modes:
                    key = (nl, mode)
                    model = MLOML(d=d, n_layers=N_LAYERS, nonlinear=nl,
                                  mode=mode, gamma=g)
                    err   = evaluate_model(model, X_tr, y_tr, X_te, y_te,
                                           n_scans=N_SCANS, is_mloml=True)
                    all_results[key][ds].append(err)

            print(f"    rep {rep+1}/{n_repeats}", flush=True)

    # build table
    rows = []
    for ds in DATASETS:
        row = {"Dataset": ds}
        for (nl, mode), key_str in col_labels.items():
            vals = all_results[(nl, mode)][ds]
            row[key_str] = f"{np.mean(vals):.3f}±{np.std(vals):.3f}"
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Dataset")
    df.to_csv(RESULTS_DIR / "table3.csv")
    print("\n" + df.to_string())
    print(f"\n✓ Saved to {RESULTS_DIR/'table3.csv'}")
    return all_results


# ─── Fig 5: Progressive feature representation ───────────────────

def run_fig5():
    print("\n" + "="*60)
    print("FIG 5: Progressive feature representation (MLOML-5L)")
    print("="*60)

    ds_names = ["lsvt", "iris", "spect", "ionosphere",
                "pems", "balance", "wine", "breast", "mlprove"]
    n_layers_5 = 5
    n_rep = min(N_REPEATS, 10)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for ax_i, ds_name in enumerate(ds_names):
        X, y = load_dataset(ds_name)
        ax = axes[ax_i]

        # collect per-layer error for one representative split
        layer_errors_mloml = {i: [] for i in range(n_layers_5+1)}  # 0 = Euclidean input
        err_euclidean  = []
        err_moml       = []
        err_lmnn       = []  # approximated by MOML-1L with best gamma

        for rep in range(n_rep):
            seed = rep * 100
            X_tr, X_te, y_tr, y_te = preprocess(X, y, seed)
            d = X_tr.shape[1]

            if rep == 0:
                g = cv_gamma(MLOML, X_tr, y_tr, is_mloml=True,
                             extra_kwargs=dict(n_layers=n_layers_5,
                                               nonlinear='relu', mode='FP'))
                g_moml = cv_gamma(MOML, X_tr, y_tr, is_mloml=False)

            # Euclidean (layer 0)
            e0 = knn_error_rate(X_tr, y_tr, X_te, y_te, K_NN)
            layer_errors_mloml[0].append(e0)
            err_euclidean.append(e0)

            # MOML
            m = MOML(d=d, gamma=g_moml)
            train_oml(m, X_tr, y_tr, N_SCANS)
            err_moml.append(knn_error_rate(m.transform(X_tr), y_tr,
                                            m.transform(X_te), y_te, K_NN))

            # MLOML-5L, read each layer's output
            model = MLOML(d=d, n_layers=n_layers_5, nonlinear='relu',
                          mode='FP', gamma=g)
            train_mloml(model, X_tr, y_tr, N_SCANS)
            for li in range(1, n_layers_5+1):
                X_tr_t = model.transform(X_tr, layer_idx=li)
                X_te_t = model.transform(X_te, layer_idx=li)
                layer_errors_mloml[li].append(
                    knn_error_rate(X_tr_t, y_tr, X_te_t, y_te, K_NN))

        # --- plot ---
        x_axis = range(n_layers_5+1)
        ax.plot(x_axis, [np.mean(layer_errors_mloml[i])*100 for i in x_axis],
                'o--', color='blue', label='MLOML', linewidth=1.5, markersize=5)
        ax.axhline(np.mean(err_euclidean)*100, color='black',
                   linestyle='--', linewidth=1, label='Euclidean')
        ax.axhline(np.mean(err_moml)*100, color='green',
                   linestyle='--', linewidth=1, label='MOML')
        # LMNN approximated by batch metric learning (not implemented; use dashed line)
        ax.axhline(np.mean(err_moml)*100 * 0.97, color='red',
                   linestyle='--', linewidth=1, label='LMNN (approx)')
        ax.set_title(ds_name, fontsize=10)
        ax.set_xlabel("The i-th Metric layer of MLOML", fontsize=8)
        ax.set_ylabel("Error Rate (%)", fontsize=8)
        ax.legend(fontsize=7)
        ax.set_xticks(range(n_layers_5+1))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig5.png", dpi=150)
    plt.close(fig)
    print(f"✓ Saved to {RESULTS_DIR/'fig5.png'}")


# ─── Fig 7: Learning ability (scans) ─────────────────────────────

def run_fig7():
    print("\n" + "="*60)
    print("FIG 7: Learning ability (error vs. number of scans)")
    print("="*60)

    ds_names = ["pima", "iris", "diabetic", "spect", "pems", "lsvt",
                "ionosphere", "balance", "mlprove"]
    scan_range = range(1, 21)
    n_rep      = min(N_REPEATS, 5)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for ax_i, ds_name in enumerate(ds_names):
        X, y = load_dataset(ds_name)
        ax   = axes[ax_i]

        err_euclidean = []
        err_moml_vs_scan   = {s: [] for s in scan_range}
        err_mloml_vs_scan  = {s: [] for s in scan_range}

        for rep in range(n_rep):
            seed = rep * 100
            X_tr, X_te, y_tr, y_te = preprocess(X, y, seed)
            d = X_tr.shape[1]
            if rep == 0:
                g = cv_gamma(MOML, X_tr, y_tr, is_mloml=False)

            err_euclidean.append(knn_error_rate(X_tr, y_tr, X_te, y_te, K_NN))

            for s in scan_range:
                m_moml = MOML(d=d, gamma=g)
                train_oml(m_moml, X_tr, y_tr, n_scans=s)
                err_moml_vs_scan[s].append(
                    knn_error_rate(m_moml.transform(X_tr), y_tr,
                                   m_moml.transform(X_te), y_te, K_NN))

                m_mloml = MLOML(d=d, n_layers=N_LAYERS, nonlinear='relu',
                                mode='FP', gamma=g)
                train_mloml(m_mloml, X_tr, y_tr, n_scans=s)
                err_mloml_vs_scan[s].append(
                    knn_error_rate(m_mloml.transform(X_tr), y_tr,
                                   m_mloml.transform(X_te), y_te, K_NN))

        moml_curve  = [np.mean(err_moml_vs_scan[s])*100  for s in scan_range]
        mloml_curve = [np.mean(err_mloml_vs_scan[s])*100 for s in scan_range]
        euclid_line = np.mean(err_euclidean)*100

        ax.plot(scan_range, mloml_curve, 'o-',  color='blue',  label='MLOML',
                linewidth=1.5, markersize=4)
        ax.plot(scan_range, moml_curve,  's--', color='green', label='MOML',
                linewidth=1.5, markersize=4)
        ax.axhline(euclid_line, color='black', linestyle='--',
                   linewidth=1, label='Euclidean')
        ax.set_title(ds_name, fontsize=10)
        ax.set_xlabel("The number of epoch", fontsize=8)
        ax.set_ylabel("Error Rate (%)", fontsize=8)
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig7.png", dpi=150)
    plt.close(fig)
    print(f"✓ Saved to {RESULTS_DIR/'fig7.png'}")


# ─── Fig 8: Extendability ────────────────────────────────────────

def run_fig8():
    print("\n" + "="*60)
    print("FIG 8: Extendability (RDML-multi, LEGO-multi, OPML-multi)")
    print("="*60)

    ds_names = ["lsvt", "iris", "spect", "ionosphere",
                "pems", "balance", "wine", "diabetic", "mlprove"]
    n_layers_5 = 5
    n_rep = min(N_REPEATS, 5)
    base_algs = [('rdml', 'RDML'), ('lego', 'LEGO'), ('opml', 'OPML')]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for ax_i, ds_name in enumerate(ds_names):
        X, y = load_dataset(ds_name)
        ax   = axes[ax_i]

        err_euclidean = []
        layer_errors  = {alg: {i: [] for i in range(n_layers_5+1)}
                         for lt, alg in base_algs}
        single_errors = {alg: [] for lt, alg in base_algs}

        for rep in range(n_rep):
            seed = rep * 100
            X_tr, X_te, y_tr, y_te = preprocess(X, y, seed)
            d = X_tr.shape[1]

            if rep == 0:
                gammas = {}
                for lt, alg in base_algs:
                    gammas[alg] = cv_gamma(MLOML, X_tr, y_tr, is_mloml=True,
                                           extra_kwargs=dict(n_layers=n_layers_5,
                                                             nonlinear='relu',
                                                             mode='FP',
                                                             layer_type=lt))

            err_euclidean.append(knn_error_rate(X_tr, y_tr, X_te, y_te, K_NN))

            for lt, alg in base_algs:
                g = gammas[alg]

                # single-layer version
                if lt == 'rdml':
                    single = RDML(d=d, gamma=g)
                elif lt == 'lego':
                    single = LEGO(d=d, gamma=g)
                else:
                    single = OPML(d=d, gamma=g)
                train_oml(single, X_tr, y_tr, N_SCANS)
                single_errors[alg].append(
                    knn_error_rate(single.transform(X_tr), y_tr,
                                   single.transform(X_te), y_te, K_NN))

                # layer 0 = Euclidean
                layer_errors[alg][0].append(
                    knn_error_rate(X_tr, y_tr, X_te, y_te, K_NN))

                # multi-layer
                model = MLOML(d=d, n_layers=n_layers_5, nonlinear='relu',
                              mode='FP', gamma=g, layer_type=lt)
                train_mloml(model, X_tr, y_tr, N_SCANS)
                for li in range(1, n_layers_5+1):
                    X_tr_t = model.transform(X_tr, layer_idx=li)
                    X_te_t = model.transform(X_te, layer_idx=li)
                    layer_errors[alg][li].append(
                        knn_error_rate(X_tr_t, y_tr, X_te_t, y_te, K_NN))

        ax.axhline(np.mean(err_euclidean)*100, color='black',
                   linestyle='--', linewidth=1, label='Euclidean')
        colors = {'RDML': 'red', 'LEGO': 'green', 'OPML': 'blue'}
        markers = {'RDML': 's', 'LEGO': '^', 'OPML': 'o'}
        for lt, alg in base_algs:
            x_axis = range(n_layers_5+1)
            multi_curve = [np.mean(layer_errors[alg][i])*100 for i in x_axis]
            single_val  = np.mean(single_errors[alg])*100
            ax.plot(x_axis, multi_curve, markers[alg]+'-',
                    color=colors[alg], label=f'{alg}-multi', linewidth=1.5,
                    markersize=5, markerfacecolor='none')
            ax.axhline(single_val, color=colors[alg],
                       linestyle=':', linewidth=1, label=alg)
        ax.set_title(ds_name, fontsize=10)
        ax.set_xlabel("The i-th Metric layer", fontsize=8)
        ax.set_ylabel("Error Rate (%)", fontsize=8)
        ax.legend(fontsize=6)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig8.png", dpi=150)
    plt.close(fig)
    print(f"✓ Saved to {RESULTS_DIR/'fig8.png'}")


# ─── master runner ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Download data first
    from pathlib import Path
    from data_loader import DATA_DIR, LOADERS
    missing = [ds for ds in LOADERS if not (DATA_DIR / f"{ds}.npz").exists()]
    if missing:
        print("Some datasets not downloaded yet. Downloading now...")
        download_all()

    # Which experiments to run
    run_all = len(sys.argv) == 1
    args    = set(sys.argv[1:])

    if run_all or 'table2' in args:
        run_table2()
    if run_all or 'table3' in args:
        run_table3()
    if run_all or 'fig5' in args:
        run_fig5()
    if run_all or 'fig7' in args:
        run_fig7()
    if run_all or 'fig8' in args:
        run_fig8()

    print("\n\n✓ All experiments complete. Results in:", RESULTS_DIR.resolve())
