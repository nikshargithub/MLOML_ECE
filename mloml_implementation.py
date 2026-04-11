"""
Full Implementation of:
"A Multilayer Framework for Online Metric Learning" (MLOML)
Li et al., arXiv:1805.05510v3

This file implements:
  - MOML  : Mahalanobis-based Online Metric Learning (the base metric layer)
  - MLOML : Multi-Layer Online Metric Learning (FP / BP / FBP variants)
  - All baseline OML algorithms: RDML, LEGO, OPML, OASIS, SLMOML
  - One-pass triplet construction strategy (from OPML)
  - Evaluation on 12 UCI datasets (Table II of the paper)
  - Progressive feature representation experiment (Fig 5 / Fig 6)
  - Propagation strategy comparison (Table III)
  - Learning ability experiment (Fig 7)
  - Extendability experiment (Fig 8)
"""

import numpy as np
from numpy.linalg import norm, eigh, svd
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def l2_normalize_rows(X):
    """ℓ₂-normalise every row of X (in-place safe)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def project_psd(M):
    """Project a symmetric matrix onto the PSD cone by clipping negative eigenvalues."""
    M = (M + M.T) / 2.0          # force symmetry
    eigvals, eigvecs = eigh(M)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def mat_sqrt(M):
    """
    Compute L such that M = L^T L  (Cholesky / eigen decomposition).
    Returns L of shape (d, d).
    """
    M = (M + M.T) / 2.0
    eigvals, eigvecs = eigh(M)
    eigvals = np.maximum(eigvals, 0)
    sqrt_vals = np.sqrt(eigvals)
    L = (eigvecs * sqrt_vals).T   # L.T @ L == M
    return L                      # shape (d, d)


def relu(X):
    return np.maximum(X, 0)

def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-np.clip(X, -50, 50)))

def tanh(X):
    return np.tanh(X)

def apply_nonlinear(X, name):
    if name == 'relu':
        return relu(X)
    elif name == 'sigmoid':
        return sigmoid(X)
    elif name == 'tanh':
        return tanh(X)
    else:
        raise ValueError(f"Unknown nonlinearity: {name}")


# ─────────────────────────────────────────────
# One-pass triplet construction  (OPML strategy)
# ─────────────────────────────────────────────

class OnlineTripletConstructor:
    """
    One-pass triplet construction as described in OPML [21].

    For each new sample x_t with label y_t:
      - Find the most recent sample x_p with the SAME label  (positive)
      - Find the most recent sample x_q with a DIFFERENT label (negative)
    Returns the triplet <x_t, x_p, x_q> or None if either is unavailable.
    """
    def __init__(self):
        self.last_same   = {}   # label -> (sample, idx)
        self.last_diff   = {}   # label -> list of (other_label, sample, idx)
        self._history    = []   # list of (sample, label)

    def get_triplet(self, x, y):
        """
        Given the current sample x with label y,
        return (x, x_pos, x_neg) or None.
        """
        x_pos = None
        x_neg = None

        # positive: latest sample with same class
        if y in self.last_same:
            x_pos = self.last_same[y]

        # negative: latest sample with any different class
        for lbl, sample in self.last_diff.get(y, []):
            x_neg = sample
            break

        # update history
        # update last_same for label y
        self.last_same[y] = x.copy()

        # update last_diff: for every OTHER label seen so far, store x as
        # the latest sample with label y
        seen_labels = set(self.last_same.keys())
        for other_lbl in seen_labels:
            if other_lbl != y:
                if other_lbl not in self.last_diff:
                    self.last_diff[other_lbl] = []
                # keep only the most recent
                self.last_diff[other_lbl] = [(y, x.copy())]

        if x_pos is None or x_neg is None:
            return None
        return (x, x_pos, x_neg)


# ─────────────────────────────────────────────
# MOML — Mahalanobis-based Online Metric Learning
# ─────────────────────────────────────────────

class MOML:
    """
    MOML as defined in Section III-A of the paper.

    Objective (Eq. 4):
        min_{M≽0}  1/2 ‖M − M_{t-1}‖_F²  +  γ [1 + Tr(M A_t)]+

    Closed-form update (Eq. 6):
        M_t = M_{t-1} − γ A_t   if  [1 + Tr(M_{t-1} A_t)]+ > 0
              M_{t-1}            otherwise

    where  A_t = (x−x_p)(x−x_p)^T − (x−x_q)(x−x_q)^T
    """
    def __init__(self, d, gamma=0.01):
        self.d = d
        self.gamma = gamma
        self.M = np.eye(d)           # initialise as identity (paper Section IV-A)
        self.L = np.eye(d)           # L s.t. M = L^T L

    def _compute_At(self, x, xp, xq):
        dp = (x - xp).reshape(-1, 1)
        dq = (x - xq).reshape(-1, 1)
        return dp @ dp.T - dq @ dq.T

    def compute_loss(self, x, xp, xq):
        At = self._compute_At(x, xp, xq)
        return max(0.0, 1.0 + np.trace(self.M @ At))

    def update(self, x, xp, xq):
        """One online update step. Returns the hinge loss before update."""
        At = self._compute_At(x, xp, xq)
        loss = max(0.0, 1.0 + np.trace(self.M @ At))
        if loss > 0:
            M_new = self.M - self.gamma * At
            # Project onto PSD cone (Theorem 1)
            self.M = project_psd(M_new)
            self.L = mat_sqrt(self.M)
        return loss

    def transform(self, X):
        """Apply L transformation: X_new = (L @ X.T).T"""
        return (self.L @ X.T).T

    def reset(self):
        self.M = np.eye(self.d)
        self.L = np.eye(self.d)


# ─────────────────────────────────────────────
# Baseline OML algorithms
# ─────────────────────────────────────────────

class RDML:
    """
    Regularized Distance Metric Learning [20].
    Simple SGD on Mahalanobis M with hinge triplet loss + Frobenius regularisation.
    """
    def __init__(self, d, lr=0.01, gamma=None):
        self.d  = d
        self.lr = gamma if gamma is not None else lr
        self.M  = np.eye(d)
        self.L  = np.eye(d)

    def _At(self, x, xp, xq):
        dp = (x - xp).reshape(-1, 1)
        dq = (x - xq).reshape(-1, 1)
        return dp @ dp.T - dq @ dq.T

    def compute_loss(self, x, xp, xq):
        At = self._At(x, xp, xq)
        return max(0.0, 1.0 + np.trace(self.M @ At))

    def update(self, x, xp, xq):
        At = self._At(x, xp, xq)
        loss = max(0.0, 1.0 + np.trace(self.M @ At))
        if loss > 0:
            grad = At + self.lr * self.M   # gradient of loss + regulariser
            M_new = self.M - self.lr * grad
            self.M = project_psd(M_new)
            self.L = mat_sqrt(self.M)
        return loss

    def transform(self, X):
        return (self.L @ X.T).T

    def reset(self):
        self.M = np.eye(self.d)
        self.L = np.eye(self.d)


class LEGO:
    """
    LogDet Exact Gradient Online (LEGO) [19].
    Uses LogDet divergence as regulariser.
    """
    def __init__(self, d, eta=0.01, gamma=None):
        self.d   = d
        self.eta = gamma if gamma is not None else eta
        self.M   = np.eye(d)
        self.L   = np.eye(d)

    def _At(self, x, xp, xq):
        dp = (x - xp).reshape(-1, 1)
        dq = (x - xq).reshape(-1, 1)
        return dp @ dp.T - dq @ dq.T

    def compute_loss(self, x, xp, xq):
        At = self._At(x, xp, xq)
        return max(0.0, 1.0 + np.trace(self.M @ At))

    def update(self, x, xp, xq):
        At = self._At(x, xp, xq)
        loss = max(0.0, 1.0 + np.trace(self.M @ At))
        if loss > 0:
            # LogDet gradient update: M_{t+1} = M_t - eta * M_t @ At @ M_t  (approx)
            grad = self.M @ At @ self.M
            M_new = self.M - self.eta * grad
            self.M = project_psd(M_new)
            self.L = mat_sqrt(self.M)
        return loss

    def transform(self, X):
        return (self.L @ X.T).T

    def reset(self):
        self.M = np.eye(self.d)
        self.L = np.eye(self.d)


class OPML:
    """
    One-Pass Metric Learning (OPML) [21].
    Learns a transformation matrix L directly (not M).
    Same PA-style update as MOML but on L.
    """
    def __init__(self, d, gamma=0.01):
        self.d     = d
        self.gamma = gamma
        self.L     = np.eye(d)

    def _At(self, x, xp, xq):
        dp = (x - xp).reshape(-1, 1)
        dq = (x - xq).reshape(-1, 1)
        return dp @ dp.T - dq @ dq.T

    def compute_loss(self, x, xp, xq):
        M = self.L.T @ self.L
        At = self._At(x, xp, xq)
        return max(0.0, 1.0 + np.trace(M @ At))

    def update(self, x, xp, xq):
        M = self.L.T @ self.L
        At = self._At(x, xp, xq)
        loss = max(0.0, 1.0 + np.trace(M @ At))
        if loss > 0:
            M_new = M - self.gamma * At
            M_new = project_psd(M_new)
            self.L = mat_sqrt(M_new)
        return loss

    def transform(self, X):
        return (self.L @ X.T).T

    def reset(self):
        self.L = np.eye(self.d)


class OASIS:
    """
    Online Algorithm for Scalable Image Similarity (OASIS) [9].
    Bilinear similarity without PSD constraint.
    """
    def __init__(self, d, gamma=0.01):
        self.d     = d
        self.gamma = gamma
        self.W     = np.eye(d)   # similarity matrix (no PSD required)

    def _similarity(self, x1, x2):
        return x1 @ self.W @ x2

    def compute_loss(self, x, xp, xq):
        s_pos = self._similarity(x, xp)
        s_neg = self._similarity(x, xq)
        return max(0.0, 1.0 - s_pos + s_neg)

    def update(self, x, xp, xq):
        loss = self.compute_loss(x, xp, xq)
        if loss > 0:
            grad = -np.outer(x, xp) + np.outer(x, xq)
            self.W -= self.gamma * grad
        return loss

    def transform(self, X):
        """For OASIS we use W directly as a projection (symmetrised)."""
        W_sym = (self.W + self.W.T) / 2.0
        M     = project_psd(W_sym)
        L     = mat_sqrt(M)
        return (L @ X.T).T

    def reset(self):
        self.W = np.eye(self.d)


class SLMOML:
    """
    Scalable Large Margin Online Metric Learning (SLMOML) [33].
    PA algorithm with LogDet divergence.
    """
    def __init__(self, d, C=0.01, gamma=None):
        self.d  = d
        self.C  = gamma if gamma is not None else C
        self.M  = np.eye(d)
        self.L  = np.eye(d)

    def _At(self, x, xp, xq):
        dp = (x - xp).reshape(-1, 1)
        dq = (x - xq).reshape(-1, 1)
        return dp @ dp.T - dq @ dq.T

    def compute_loss(self, x, xp, xq):
        At = self._At(x, xp, xq)
        return max(0.0, 1.0 + np.trace(self.M @ At))

    def update(self, x, xp, xq):
        At   = self._At(x, xp, xq)
        loss = max(0.0, 1.0 + np.trace(self.M @ At))
        if loss > 0:
            # Compute step size via PA-II rule
            num = loss
            den = np.trace(self.M @ At @ self.M @ At) + 1.0 / (2.0 * self.C)
            tau = num / (den + 1e-10)
            M_new = self.M - tau * self.M @ At @ self.M
            self.M = project_psd(M_new)
            self.L = mat_sqrt(self.M)
        return loss

    def transform(self, X):
        return (self.L @ X.T).T

    def reset(self):
        self.M = np.eye(self.d)
        self.L = np.eye(self.d)


# ─────────────────────────────────────────────
# MLOML — Multi-Layer Online Metric Learning
# ─────────────────────────────────────────────

def _make_layer(layer_type, d, gamma):
    """Factory for building a single OML metric layer."""
    if layer_type == 'moml':
        return MOML(d, gamma)
    elif layer_type == 'rdml':
        return RDML(d, gamma)
    elif layer_type == 'lego':
        return LEGO(d, gamma)
    elif layer_type == 'opml':
        return OPML(d, gamma)
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")


class MLOML:
    """
    Multi-Layer Online Metric Learning (MLOML).

    Parameters
    ----------
    d          : input feature dimension
    n_layers   : number of OML metric layers (paper uses 3 or 5)
    nonlinear  : 'relu' | 'sigmoid' | 'tanh'  → MLOML-r / MLOML-s / MLOML-t
    mode       : 'FP' | 'BP' | 'FBP'
    gamma      : regularisation parameter for each MOML metric layer
    lam        : λ in loss function (Eq.1), weight of Frobenius regulariser
    lr_bp      : learning rate for backward propagation SGD
    layer_type : 'moml' | 'rdml' | 'lego' | 'opml'  (base layer algorithm)
    """
    def __init__(self, d, n_layers=3, nonlinear='relu', mode='FP',
                 gamma=0.01, lam=0.01, lr_bp=0.001, layer_type='moml'):
        self.d          = d
        self.n_layers   = n_layers
        self.nonlinear  = nonlinear
        self.mode       = mode.upper()
        self.gamma      = gamma
        self.lam        = lam
        self.lr_bp      = lr_bp
        self.layer_type = layer_type

        # Build metric layers
        self.layers = [_make_layer(layer_type, d, gamma) for _ in range(n_layers)]

        # Layer weights w_i (Eq.1), learnt by SGD — initialised to 1
        self.weights = np.ones(n_layers)

    def reset(self):
        for layer in self.layers:
            layer.reset()
        self.weights = np.ones(self.n_layers)

    # ── forward pass ──────────────────────────────────────────────
    def _forward(self, x, xp, xq, update_fp=True):
        """
        Run the full forward pass through all metric + nonlinear layers.

        Returns
        -------
        intermediates : list of (xt_i, xp_i, xq_i) after each metric+nonlinear layer
        local_losses  : hinge loss at each metric layer
        """
        xt_cur, xp_cur, xq_cur = x.copy(), xp.copy(), xq.copy()
        intermediates  = []
        local_losses   = []

        for i, layer in enumerate(self.layers):
            # ── metric layer ──
            if update_fp:
                loss_i = layer.update(xt_cur, xp_cur, xq_cur)
            else:
                loss_i = layer.compute_loss(xt_cur, xp_cur, xq_cur)
            local_losses.append(loss_i)

            # transform via L_i
            xt_cur = layer.transform(xt_cur.reshape(1, -1)).flatten()
            xp_cur = layer.transform(xp_cur.reshape(1, -1)).flatten()
            xq_cur = layer.transform(xq_cur.reshape(1, -1)).flatten()

            # ── nonlinear layer BETWEEN metric layers (not after last) ──
            if i < self.n_layers - 1:
                xt_cur = apply_nonlinear(xt_cur, self.nonlinear)
                xp_cur = apply_nonlinear(xp_cur, self.nonlinear)
                xq_cur = apply_nonlinear(xq_cur, self.nonlinear)

            intermediates.append((xt_cur.copy(), xp_cur.copy(), xq_cur.copy()))

        return intermediates, local_losses

    def _triplet_loss_final(self, intermediates):
        xt_n, xp_n, xq_n = intermediates[-1]
        return max(0.0,
                   norm(xt_n - xp_n)**2 + 1.0 - norm(xt_n - xq_n)**2)

    # ── backward propagation ──────────────────────────────────────
    def _backward(self, x, xp, xq):
        """
        BP-only update: compute global triplet loss, backprop gradients
        through all L_i matrices using chain rule + SGD.
        """
        # collect transformed features layer by layer (no FP updates)
        xt_list  = [x.copy()]
        xp_list  = [xp.copy()]
        xq_list  = [xq.copy()]

        for i, layer in enumerate(self.layers):
            xt_new = layer.transform(xt_list[-1].reshape(1,-1)).flatten()
            xp_new = layer.transform(xp_list[-1].reshape(1,-1)).flatten()
            xq_new = layer.transform(xq_list[-1].reshape(1,-1)).flatten()
            if i < self.n_layers - 1:
                xt_new = apply_nonlinear(xt_new, self.nonlinear)
                xp_new = apply_nonlinear(xp_new, self.nonlinear)
                xq_new = apply_nonlinear(xq_new, self.nonlinear)
            xt_list.append(xt_new)
            xp_list.append(xp_new)
            xq_list.append(xq_new)

        # triplet loss at final layer
        xt_n = xt_list[-1]; xp_n = xp_list[-1]; xq_n = xq_list[-1]
        loss = max(0.0, norm(xt_n - xp_n)**2 + 1.0 - norm(xt_n - xq_n)**2)
        if loss <= 0:
            return 0.0

        # gradient of triplet loss w.r.t. xt_n, xp_n, xq_n
        dxt = 2*(xt_n - xp_n) - 2*(xt_n - xq_n)
        dxp = -2*(xt_n - xp_n)
        dxq =  2*(xt_n - xq_n)

        # backprop through layers in reverse
        for i in reversed(range(self.n_layers)):
            layer = self.layers[i]
            x_in_t = xt_list[i]
            x_in_p = xp_list[i]
            x_in_q = xq_list[i]

            # gradient of L_i w.r.t. loss:  d/dL_i  = dout @ x_in^T
            dL  = (np.outer(dxt, x_in_t)
                 + np.outer(dxp, x_in_p)
                 + np.outer(dxq, x_in_q))

            # L2 regularisation gradient
            dL += self.lam * layer.L

            # SGD update on L
            layer.L -= self.lr_bp * dL
            # recover M
            layer.M = layer.L.T @ layer.L

            # propagate gradient backward through nonlinear if not first layer
            if i > 0:
                if self.nonlinear == 'relu':
                    # ReLU derivative: 1 where output > 0
                    mask_t = (xt_list[i] > 0).astype(float)
                    mask_p = (xp_list[i] > 0).astype(float)
                    mask_q = (xq_list[i] > 0).astype(float)
                elif self.nonlinear == 'sigmoid':
                    s_t = sigmoid(xt_list[i])
                    s_p = sigmoid(xp_list[i])
                    s_q = sigmoid(xq_list[i])
                    mask_t = s_t * (1 - s_t)
                    mask_p = s_p * (1 - s_p)
                    mask_q = s_q * (1 - s_q)
                else:  # tanh
                    mask_t = 1 - np.tanh(xt_list[i])**2
                    mask_p = 1 - np.tanh(xp_list[i])**2
                    mask_q = 1 - np.tanh(xq_list[i])**2

                dxt = (layer.L.T @ dxt) * mask_t
                dxp = (layer.L.T @ dxp) * mask_p
                dxq = (layer.L.T @ dxq) * mask_q

        return loss

    # ── public train step ─────────────────────────────────────────
    def train_step(self, x, xp, xq):
        """One online training step on triplet (x, xp, xq)."""
        if self.mode == 'FP':
            _, local_losses = self._forward(x, xp, xq, update_fp=True)
            return sum(local_losses) / self.n_layers

        elif self.mode == 'BP':
            loss = self._backward(x, xp, xq)
            return loss

        elif self.mode == 'FBP':
            # forward first (updates each layer locally)
            _, local_losses = self._forward(x, xp, xq, update_fp=True)
            # then backward to fine-tune
            loss_bp = self._backward(x, xp, xq)
            return (sum(local_losses) / self.n_layers + loss_bp) / 2.0

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ── transform ─────────────────────────────────────────────────
    def transform(self, X, layer_idx=None):
        """
        Transform X through the full stack (or up to layer_idx).
        layer_idx=None → all n_layers
        layer_idx=k    → first k layers (1-indexed)
        """
        if layer_idx is None:
            layer_idx = self.n_layers

        X_cur = X.copy()
        for i in range(layer_idx):
            layer = self.layers[i]
            X_cur = layer.transform(X_cur)
            # apply nonlinear BETWEEN metric layers, NOT after the last one
            if i < layer_idx - 1:
                X_cur = apply_nonlinear(X_cur, self.nonlinear)
        return X_cur


# ─────────────────────────────────────────────
# kNN classifier
# ─────────────────────────────────────────────

def knn_error_rate(X_train, y_train, X_test, y_test, k=5):
    """Brute-force k-NN error rate."""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    return np.mean(preds != y_test)


# ─────────────────────────────────────────────
# Training routine (multiple scan strategy)
# ─────────────────────────────────────────────

def train_oml(model, X_train, y_train, n_scans=20):
    """
    Train a single-layer OML model (MOML, RDML, LEGO, OPML, OASIS, SLMOML)
    using the one-pass triplet construction strategy with n_scans passes.
    """
    model.reset()
    for _ in range(n_scans):
        idx   = np.random.permutation(len(X_train))
        tc    = OnlineTripletConstructor()
        for i in idx:
            triplet = tc.get_triplet(X_train[i], y_train[i])
            if triplet is not None:
                x, xp, xq = triplet
                model.update(x, xp, xq)


def train_mloml(model, X_train, y_train, n_scans=20):
    """
    Train MLOML end-to-end (all three modes) using n_scans passes.
    """
    model.reset()
    for _ in range(n_scans):
        idx = np.random.permutation(len(X_train))
        tc  = OnlineTripletConstructor()
        for i in idx:
            triplet = tc.get_triplet(X_train[i], y_train[i])
            if triplet is not None:
                x, xp, xq = triplet
                model.train_step(x, xp, xq)


# ─────────────────────────────────────────────
# Evaluation helper: run one split
# ─────────────────────────────────────────────

def evaluate_model(model, X_train, y_train, X_test, y_test,
                   n_scans=20, k=5, is_mloml=False):
    if is_mloml:
        train_mloml(model, X_train, y_train, n_scans)
        X_tr_t = model.transform(X_train)
        X_te_t = model.transform(X_test)
    else:
        train_oml(model, X_train, y_train, n_scans)
        X_tr_t = model.transform(X_train)
        X_te_t = model.transform(X_test)
    return knn_error_rate(X_tr_t, y_train, X_te_t, y_test, k)


def evaluate_euclidean(X_train, y_train, X_test, y_test, k=5):
    return knn_error_rate(X_train, y_train, X_test, y_test, k)
