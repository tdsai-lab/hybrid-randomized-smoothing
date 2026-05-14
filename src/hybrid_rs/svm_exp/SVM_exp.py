#!/usr/bin/env python3
# ============================================================
# TRUE Hybrid RS on Tabular data (categorical L0 + numeric L2)
# - Train Linear SVM on Adult (OpenML)
# - Empirical hybrid attack at test time
# - Hybrid RS certification using the project's theory utilities:
#     certify_r_hybrid + grouped_uniform/grouped_absorb + CP-LCB
# ============================================================

# ============================================================
# Task and Dataset Description (for experiments section)
# ============================================================

# Task:
# -----
# We consider a binary classification task that plays the role of a
# *decision or safety filter*. The classifier outputs a hard decision
# y ∈ {0,1}, and robustness is defined as the invariance of this decision
# under bounded test-time perturbations.
#
# The goal of the experiment is NOT to maximize accuracy, but to:
#   (1) evaluate empirical robustness under hybrid attacks,
#   (2) provide certified robustness guarantees under joint perturbations.
#
# This mirrors the role of a safety filter in LLM systems:
#   - the output is a binary decision,
#   - the adversary can manipulate heterogeneous inputs,
#   - we want provable stability of the decision.


# Dataset:
# --------
# We use the Adult (Census Income) dataset from UCI / OpenML.
# Each example corresponds to an individual from the US census.
#
# The prediction task is:
#   y = 1  if annual income > $50K
#   y = 0  otherwise
#
# This dataset is a standard benchmark in classical ML and is well suited
# for studying robustness in mixed continuous / discrete settings.


# Input structure:
# ----------------
# Each data point is described by 14 features of two different types.
#
# (A) Continuous features (6):
#     - age
#     - fnlwgt
#     - education-num
#     - capital-gain
#     - capital-loss
#     - hours-per-week
#
# These features define a continuous Euclidean subspace. Perturbations on
# this part are measured using an L2 norm.
#
# (B) Categorical features (8):
#     - workclass
#     - education
#     - marital-status
#     - occupation
#     - relationship
#     - race
#     - sex
#     - native-country
#
# Each categorical feature has its own arity (from 2 up to 40+ values).
# They are encoded using one-hot vectors for training the classifier.
#
# Importantly, robustness on categorical inputs is measured in *grouped L0*:
# changing the value of one categorical field counts as one discrete change,
# regardless of the one-hot dimensionality.


# Model:
# ------
# The classifier is a linear SVM trained on:
#   - standardized continuous features,
#   - one-hot encoded categorical features.
#
# Although the resulting feature vector has high dimensionality
# (≈108 dimensions), the semantic structure remains:
#   - 6 continuous degrees of freedom,
#   - 8 discrete groups.
#
# This makes the setting directly comparable to hybrid
# (continuous + discrete) robustness in multimodal models.


# Threat model:
# -------------
# We consider test-time adversarial perturbations acting jointly on both
# parts of the input:
#
#   - Continuous perturbation:
#       ||δ_num||_2 ≤ r
#
#   - Discrete perturbation:
#       ||δ_cat||_0 ≤ d
#     where d counts the number of categorical fields modified.
#
# This hybrid threat model is the tabular analogue of:
#   - L2 perturbations on images,
#   - L0 token edits on text.


# Certification objective:
# ------------------------
# Using Hybrid Randomized Smoothing, we certify that the classifier's
# prediction remains unchanged for all perturbations satisfying:
#
#   ||δ_cat||_0 ≤ d  and  ||δ_num||_2 ≤ r(d)
#
# where r(d) is a certified radius that decreases as the discrete budget d
# increases.
#
# This experiment serves as a classical ML proof-of-concept for certified
# hybrid robustness, before scaling to multimodal LLM safety filters.
# ============================================================


import argparse
import random
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from sklearn.base import clone

import itertools
import math
import torch


from hybrid_rs.utils_hybrid import (
    certify_r_hybrid,
    # clopper_pearson_lcb,
    grouped_absorb,
)
from statsmodels.stats.proportion import proportion_confint




def clopper_pearson_lcb(n: int, nA: int, alpha: float) -> float:
    """
    One-sided lower confidence bound on Binomial proportion p with confidence 1-alpha.
    Uses two-sided beta interval with alpha' = 2*alpha.
    """
    n = int(n)
    nA = int(nA)
    if n <= 0:
        return float("nan")
    if nA < 0 or nA > n:
        raise ValueError(f"Invalid counts: nA={nA} must satisfy 0 <= nA <= n={n}")
    return float(proportion_confint(nA, n, alpha=2 * alpha, method="beta")[0])


# ----------------------------
# Dataset: Adult (mixed tabular)
# ----------------------------
def load_adult(seed: int = 0, test_size: float = 0.2):
    ds = fetch_openml(name="adult", version=2, as_frame=True)
    X = ds.data
    y_raw = ds.target
    y = (y_raw == ">50K").astype(int).to_numpy()

    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, num_cols, cat_cols


def build_pipeline(num_cols, cat_cols, C: float):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    svm = LinearSVC(C=C, dual=False, max_iter=4000)
    return Pipeline([("pre", pre), ("svm", svm)])


def augment_train_tabular(
    pipe,
    Xtr: pd.DataFrame,
    ytr,
    num_cols,
    cat_cols,
    K: int,
    sigma: float,
    beta: float,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Data augmentation matching the Hybrid RS kernels:
      - numeric: add N(0, sigma^2 I) in STANDARDIZED space
      - categorical: with prob beta, replace by a uniformly sampled *different* category

    Returns X_aug, y_aug with size (K+1)*n.
    """
    rng = np.random.default_rng(seed)
    ytr = np.asarray(ytr)

    # Fit preprocessor ONCE to get scaler + categories (no leakage: uses only Xtr)
    pre = pipe.named_steps["pre"]
    pre.fit(Xtr)

    num_tr = pre.named_transformers_.get("num", None)
    cat_tr = pre.named_transformers_.get("cat", None)

    if (len(num_cols) > 0) and (num_tr is None):
        raise ValueError("Missing 'num' transformer in preprocessor.")
    if (len(cat_cols) > 0) and (cat_tr is None):
        raise ValueError("Missing 'cat' transformer in preprocessor.")
    if (len(cat_cols) > 0) and (not hasattr(cat_tr, "categories_")):
        raise ValueError("Expected 'cat' transformer to be a fitted OneHotEncoder with categories_.")

    cat_categories = list(cat_tr.categories_) if len(cat_cols) > 0 else []

    # Base copy (clean)
    # X_list = [Xtr.copy(deep=True)]
    X_clean = Xtr.copy(deep=True)
    X_clean.loc[:, num_cols] = X_clean.loc[:, num_cols].astype(np.float64)
    
    
    X_clean = Xtr.copy(deep=True)
    X_clean[num_cols] = X_clean[num_cols].to_numpy(dtype=np.float64)
    X_list = [X_clean]

    # X_list = [X_clean]
    y_list = [ytr.copy()]

    n = len(Xtr)

    for k in range(K):
        Xk = Xtr.copy(deep=True)
        Xk[num_cols] = Xk[num_cols].to_numpy(dtype=np.float64)

        Xk.loc[:, num_cols] = Xk.loc[:, num_cols].astype(np.float64) # ensure float64 for precision


        # ---- numeric Gaussian in standardized space ----
        if sigma > 0 and len(num_cols) > 0:
            X_num = Xk[num_cols]
            z = num_tr.transform(X_num)                      # (n, num_dim)
            z = z + rng.normal(0.0, sigma, size=z.shape)     # add noise in standardized space
            
            X_num_noisy = num_tr.inverse_transform(z)        # back to raw
            Xk.loc[:, num_cols] = X_num_noisy.astype(np.float64)

        # ---- categorical uniform corruption ----
        if beta > 0 and len(cat_cols) > 0:
            for j, col in enumerate(cat_cols):
                cats = np.asarray(cat_categories[j], dtype=object)
                if cats.size <= 1:
                    continue

                mask = rng.random(n) < beta
                if not np.any(mask):
                    continue

                cur = Xk.loc[mask, col].to_numpy(dtype=object)
                new = cur.copy()

                # resample until different (expected very few loops)
                for _ in range(10):
                    prop = rng.choice(cats, size=new.shape[0], replace=True)
                    diff = prop != new
                    new[diff] = prop[diff]
                    if np.all(new != cur):
                        break

                # if still some equal (degenerate), force by picking an alternative deterministically
                eq = (new == cur)
                if np.any(eq):
                    # pick first category that differs
                    alt = cats[0]
                    new[eq] = alt
                    still_eq = (new == cur)
                    if np.any(still_eq):
                        new[still_eq] = cats[-1]

                Xk.loc[mask, col] = new

        X_list.append(Xk)
        y_list.append(ytr.copy())

    X_aug = pd.concat(X_list, axis=0, ignore_index=True)
    y_aug = np.concatenate(y_list, axis=0)
    return X_aug, y_aug





# ----------------------------
# Helpers in transformed space
# ----------------------------
def margin_linear_svm(pipe: Pipeline, x_dense: np.ndarray) -> float:
    svm: LinearSVC = pipe.named_steps["svm"]
    w = svm.coef_.reshape(-1)
    b = float(svm.intercept_.reshape(-1)[0])
    return float(np.dot(w, x_dense) + b)


def pred_from_margin(m: float) -> int:
    return 1 if m >= 0.0 else 0


# def add_gaussian_noise_numeric(x_dense: np.ndarray, num_dim: int, sigma: float, rng: np.random.Generator):
#     out = x_dense.copy()
#     out[:num_dim] = out[:num_dim] + rng.normal(loc=0.0, scale=sigma, size=num_dim)
#     return out

def get_num_idx(pre: ColumnTransformer) -> np.ndarray:
    feat_names = pre.get_feature_names_out()
    return np.array([i for i, n in enumerate(feat_names) if n.startswith("num__")], dtype=int)

def add_gaussian_noise_numeric(x_dense: np.ndarray, num_idx: np.ndarray, sigma: float, rng: np.random.Generator):
    out = x_dense.copy()
    out[num_idx] = out[num_idx] + rng.normal(0.0, sigma, size=len(num_idx))
    return out


def cat_group_slices(pipe: Pipeline, num_dim: int):
    pre: ColumnTransformer = pipe.named_steps["pre"]
    ohe: OneHotEncoder = pre.named_transformers_["cat"]
    sizes = [len(cats) for cats in ohe.categories_]
    sls = []
    off = num_dim
    for s in sizes:
        sls.append(slice(off, off + s))
        off += s
    return sls, sizes


# ----------------------------
# Noise on categorical groups (uniform / absorb-like)
# ----------------------------
def noise_cat_uniform(x_dense: np.ndarray, cat_slices, beta: float, rng: np.random.Generator):
    """
    Uniform replacement excluding the original category (matches grouped_uniform):
      with prob (1-beta): keep original
      with prob beta: replace uniformly among the (k-1) other categories
    """
    out = x_dense.copy()
    for sl in cat_slices:
        if rng.random() < beta:
            k = sl.stop - sl.start

            # determine current category (assume one-hot; if all zeros, pick 0 as default)
            cur = int(np.argmax(out[sl])) if float(np.sum(out[sl])) > 0.0 else 0

            if k <= 1:
                continue  # degenerate, should not happen
            if k == 2:
                j = 1 - cur
            else:
                r = int(rng.integers(low=0, high=k - 1))
                j = r if r < cur else r + 1  # skip cur

            out[sl] = 0.0
            out[sl.start + j] = 1.0

    return out


def noise_cat_absorb(x_dense: np.ndarray, cat_slices, beta: float, rng: np.random.Generator):
    """
    'Absorb' analogue: set the group to all-zeros with prob beta.
    This is a valid discrete corruption kernel on one-hot blocks.
    Your grouped_absorb should correspond to an absorbing token.
    """
    out = x_dense.copy()
    for sl in cat_slices:
        if rng.random() < beta:
            out[sl] = 0.0
    return out



def attack_hybrid_tabular_alternating(
    pipe: Pipeline,
    x_row: pd.Series,
    num_cols,
    cat_cols,
    eps_l2: float,
    steps: int,
    step: float,
    d_cat: int,
    seed: int,
    steps_per_round: int = 5,
    recompute_label_each_round: bool = True,
    log: bool = False,
    y_true: int = 0,
):
    """
    Stronger, correct alternating hybrid attack in TABULAR SPACE.

    Key fix vs previous version:
      - numeric PGD direction depends on current predicted label y_hat:
            minimize margin if y_hat=1, maximize margin if y_hat=0
        i.e. step direction is -sign(y_hat)*w_num where sign(1)=+1, sign(0)=-1.

    Alternation:
      - do 'steps_per_round' numeric PGD steps (projected to L2 ball in standardized space)
      - then do 1 greedy categorical flip (joint greedy, recomputed)
      - repeat until numeric steps and/or categorical flips budgets are exhausted

    Returns:
      x_adv: pd.Series in original tabular schema
    """
    rng = np.random.default_rng(seed)

    pre = pipe.named_steps["pre"]
    svm: LinearSVC = pipe.named_steps["svm"]
    w = svm.coef_.reshape(-1)
    b = float(svm.intercept_.reshape(-1)[0]) if hasattr(svm, "intercept_") else 0.0

    num_tr = pre.named_transformers_.get("num", None)
    cat_tr = pre.named_transformers_.get("cat", None)
    if num_tr is None or cat_tr is None:
        raise ValueError("Expected pre to have named transformers 'num' and 'cat' (ColumnTransformer).")

    if len(cat_cols) > 0:
        if not hasattr(cat_tr, "categories_"):
            raise ValueError("Expected 'cat' transformer to be a fitted OneHotEncoder with categories_.")
        cat_categories = cat_tr.categories_
        if len(cat_categories) != len(cat_cols):
            raise ValueError("Mismatch between cat_cols and OneHotEncoder.categories_ lengths.")
    else:
        cat_categories = []

    num_dim = len(num_cols)

    def margin_on_row(row: pd.Series) -> float:
        xd = pre.transform(row.to_frame().T)[0]
        return float(np.dot(w, xd) + b)

    def yhat_on_row(row: pd.Series) -> int:
        return int(pipe.predict(row.to_frame().T)[0])

    # working copy in raw space
    x_adv = x_row.copy(deep=True)

    # keep numeric state in standardized space (exact projection)
    if eps_l2 > 0 and num_dim > 0:
        x_num_raw_df = pd.DataFrame([[x_adv[c] for c in num_cols]], columns=list(num_cols))
        x_num_std0 = num_tr.transform(x_num_raw_df)[0]
        x_num_std = x_num_std0.copy()

        # small random init inside the ball
        u = rng.normal(size=num_dim)
        u = u / (np.linalg.norm(u) + 1e-12)
        x_num_std = x_num_std + 0.1 * eps_l2 * u
    else:
        x_num_std0 = None
        x_num_std = None

    # constant direction in standardized numeric coords is w_num
    # w_num = w[:num_dim].copy() if num_dim > 0 else None
    
    # --- replace this ---
    # w_num = w[:num_dim].copy() if num_dim > 0 else None

    # --- by this ---
    feat_names = pre.get_feature_names_out()
    num_idx = np.array([i for i, n in enumerate(feat_names) if n.startswith("num__")], dtype=int)
    assert len(num_idx) == num_dim, (len(num_idx), num_dim)  # sanity
    w_num = w[num_idx].copy()


    used_cols = set()

    flips_done = 0
    steps_done = 0

    if log:
        print(f"[attack] start: y_hat={yhat_on_row(x_adv)} margin={margin_on_row(x_adv):.6f}")
        
        
    sign_true = +1.0 if y_true == 1 else -1.0


    while (steps_done < steps) or (flips_done < d_cat):
        # ------------------------------------------------------------
        # (A) numeric PGD chunk
        # ------------------------------------------------------------
        if eps_l2 > 0 and num_dim > 0 and steps_done < steps:
            t = min(int(steps_per_round), int(steps - steps_done))

            # choose direction based on current predicted label (optionally recompute)
            if recompute_label_each_round:
                y_hat = yhat_on_row(x_adv)
            else:
                y_hat = yhat_on_row(x_row)

            # sign = +1.0 if y_hat == 1 else -1.0  # sign(y_hat) with y in {0,1}
            g = sign_true * w_num                      # attack direction depends on label
            g = g / (np.linalg.norm(g) + 1e-12)

            for _ in range(t):
                # move to reduce the current class confidence:
                # if y_hat=1 => reduce margin (move -w)
                # if y_hat=0 => increase margin (move +w), i.e. -sign*w with sign=-1
                x_num_std = x_num_std - float(step) * g

                # project to L2 ball around x_num_std0
                delta = x_num_std - x_num_std0
                dn = np.linalg.norm(delta)
                if dn > eps_l2:
                    x_num_std = x_num_std0 + (eps_l2 / (dn + 1e-12)) * delta

            steps_done += t

            # write back numeric part to raw row
            x_num_raw_adv = num_tr.inverse_transform(
                pd.DataFrame(x_num_std.reshape(1, -1), columns=list(num_cols))
            )[0]
            for i, c in enumerate(num_cols):
                x_adv[c] = float(x_num_raw_adv[i])

        # ------------------------------------------------------------
        # (B) one joint greedy categorical flip (recompute each step)
        # ------------------------------------------------------------
        
        if flips_done < d_cat and len(cat_cols) > 0:
            base_m = margin_on_row(x_adv)
            best_impr = 0.0
            best_col = None
            best_val = None

            for j, col in enumerate(cat_cols):
                if col in used_cols:
                    continue
                cur_val = x_adv[col]

                for v in cat_categories[j]:
                    if v == cur_val:
                        continue

                    row_t = x_adv.copy(deep=True)
                    row_t[col] = v
                    m = margin_on_row(row_t)
                    # impr = base_m - m  # want to DECREASE margin
                    impr = sign_true * (base_m - m)  # adjust for current label direction

                    if impr > best_impr:
                        best_impr = impr
                        best_col = col
                        best_val = v

            if best_col is None:
                # no improving flip exists under current state
                flips_done = d_cat
            else:
                x_adv[best_col] = best_val
                used_cols.add(best_col)
                flips_done += 1

        if log:
            print(
                f"[attack] steps={steps_done}/{steps} flips={flips_done}/{d_cat} "
                f"y_hat={yhat_on_row(x_adv)} margin={margin_on_row(x_adv):.6f}"
            )

        if (steps_done >= steps) and (flips_done >= d_cat):
            break

    return x_adv


def grouped_uniform_hetero(V_sub, beta: float, device="cpu"):
    """
    Exact heterogeneous-arity analogue of grouped_uniform(d, beta, V).

    Assumption matches your grouped_uniform():
      - with prob (1-beta): keep original token
      - with prob beta: replace uniformly among (V_j - 1) tokens excluding original
    and x and x' differ on each of these d groups.
    """
    beta_bar = 1.0 - float(beta)

    # per group: outcomes {A=z==x, B=z==x', C=z==other}
    per_group = []
    for Vj in V_sub:
        Vj = int(Vj)
        assert Vj >= 2, f"Need V_j>=2, got {Vj}"
        alpha_j = float(beta) / float(Vj - 1)
        p_other = float(beta) - alpha_j  # = beta*(Vj-2)/(Vj-1)
        per_group.append({
            "pc": (beta_bar, alpha_j, p_other),  # under x
            "pa": (alpha_j, beta_bar, p_other),  # under x'
        })

    pc_list, pa_list, g_list = [], [], []

    # enumerate 3^d joint outcomes
    for outcomes in itertools.product([0, 1, 2], repeat=len(V_sub)):
        pc = 1.0
        pa = 1.0
        for j, o in enumerate(outcomes):
            pc *= per_group[j]["pc"][o]
            pa *= per_group[j]["pa"][o]
        pc_list.append(pc)
        pa_list.append(pa)
        g_list.append(0.0 if pc == 0.0 else pa / pc)

    return (
        torch.tensor(pc_list, dtype=torch.float64, device=device),
        torch.tensor(pa_list, dtype=torch.float64, device=device),
        torch.tensor(g_list, dtype=torch.float64, device=device),
    )


def certify_example(pipe: Pipeline, x_row, num_cols, kernel: str, beta: float, sigma: float,
                    tau: float, alpha: float, n: int, batch_size: int, r_max: float, seed: int,
                    device: str = "cpu", d_max_report: int = 3):
    rng = np.random.default_rng(seed)
    pre = pipe.named_steps["pre"]

    num_idx = get_num_idx(pre)

    x0 = pre.transform(x_row.to_frame().T)[0]
    num_dim = len(num_cols)
    cat_sls, cat_sizes = cat_group_slices(pipe, num_dim)
    m = len(cat_sls)

    # Monte Carlo for pA (event = predicted label equals base prediction)
    y0 = pred_from_margin(margin_linear_svm(pipe, x0))

    nA = 0
    done = 0
    while done < n:
        b = min(batch_size, n - done)
        for _ in range(b):
            x = x0.copy()

            # categorical noise (must match the discrete kernel assumptions)
            if kernel == "absorb":
                x = noise_cat_absorb(x, cat_sls, beta, rng)
            else:
                x = noise_cat_uniform(x, cat_sls, beta, rng)

            # numeric Gaussian
            x = add_gaussian_noise_numeric(x, num_idx, sigma, rng)

            y = pred_from_margin(margin_linear_svm(pipe, x))
            if y == y0:
                nA += 1
        done += b

    pA = clopper_pearson_lcb(n, nA, alpha)

    # Hybrid certificate curve (worst-case over which categorical groups are changed)
    certified = []
    V_worst_last = None

    d_max_report = 3
    for d in range(1, min(m, d_max_report) + 1):
        if kernel == "absorb":
            pc, pa, g = grouped_absorb(d, beta, device=device)
            r = certify_r_hybrid(pA, tau, sigma, pc, pa, g, r_max)
            if r <= 0:
                break
            certified.append((d, float(r)))
            continue

        # kernel == "uniform": heterogeneous arities, take worst-case over subsets
        r_worst = float("inf")
        V_worst = None

        for idxs in itertools.combinations(range(m), d):
            V_sub = [cat_sizes[i] for i in idxs]  # true arities for those groups
            pc, pa, g = grouped_uniform_hetero(V_sub, beta, device=device)
            r_sub = certify_r_hybrid(pA, tau, sigma, pc, pa, g, r_max)
            if r_sub < r_worst:
                r_worst = float(r_sub)
                V_worst = max(int(v) for v in V_sub)

        if (not math.isfinite(r_worst)) or (r_worst <= 0.0):
            break

        certified.append((d, float(r_worst)))
        if r_worst < 1E-6:
            break
        V_worst_last = V_worst

    return {
        "y0": int(y0),
        "pA": float(pA),
        "nA": int(nA),
        "V_worst_last": (None if V_worst_last is None else int(V_worst_last)),
        "certified": certified,
    }



# ------------------------------------------------------------
# Smoothed prediction (Hybrid RS) centered at a *tabular row*
# ------------------------------------------------------------
def predict_hybrid_rs(
    pipe,
    x_row: pd.Series,
    num_cols,
    kernel: str = "uniform",
    beta: float = 0.25,
    sigma: float = 0.5,
    n: int = 2000,
    batch_size: int = 100,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Smoothed prediction under HYBRID RS noise (categorical + numeric),
    centered at a point in the *original tabular space*.

    This function is the one you want for the paper: it takes a valid
    tabular row (continuous + categorical), then evaluates the model
    through the preprocessing pipeline, and performs a majority vote
    under the joint noise model.
    """
    rng = np.random.default_rng(seed)
    pre = pipe.named_steps["pre"]
    num_idx = get_num_idx(pre)

    # start from the transformed representation (model input space)
    x0 = pre.transform(x_row.to_frame().T)[0]
    num_dim = len(num_cols)
    cat_sls, _ = cat_group_slices(pipe, num_dim)

    y_base = pred_from_margin(margin_linear_svm(pipe, x0))

    counts = {0: 0, 1: 0}
    done = 0
    while done < n:
        b = min(batch_size, n - done)
        for _ in range(b):
            x = x0.copy()

            # categorical noise (in transformed space but group-wise, consistent with kernel)
            if kernel == "absorb":
                x = noise_cat_absorb(x, cat_sls, beta, rng)
            else:
                x = noise_cat_uniform(x, cat_sls, beta, rng)

            # numeric Gaussian (on the first num_dim coords)
            x = add_gaussian_noise_numeric(x, num_idx, sigma, rng)

            y = pred_from_margin(margin_linear_svm(pipe, x))
            counts[int(y)] += 1
        done += b

    y_smooth = 1 if counts[1] >= counts[0] else 0
    p_hat = counts[y_smooth] / float(n)

    return {
        "y_base": int(y_base),
        "y_smooth": int(y_smooth),
        "p_hat": float(p_hat),
        "counts": counts,
        "n": int(n),
    }


# ------------------------------------------------------------
# Evaluate smoothed prediction under a *tabular-space attack*
# ------------------------------------------------------------
def eval_under_attack_with_smoothed(
    pipe,
    x_row: pd.Series,
    y_true: int,
    num_cols,
    cat_cols,
    # attack params (tabular-space)
    attack_eps_l2: float = 1.0,
    attack_steps: int = 30,
    attack_step: float = 0.2,
    attack_d_cat: int = 2,
    # smoothing params
    kernel: str = "uniform",
    beta: float = 0.25,
    sigma: float = 0.5,
    n_smooth: int = 2000,
    batch_size: int = 100,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Correct evaluation protocol:
      - build an adversarial example in the *original tabular space*
        (continuous edits + categorical field swaps),
      - then run the smoothed predictor predict_hybrid_rs on both x and x_adv.

    Requirement:
      attack_hybrid_tabular(...) must return a *tabular row* x_adv_row (pd.Series or 1-row DataFrame),
      NOT a dense transformed vector.
    """
    # smoothed + base prediction on clean
    res_clean = predict_hybrid_rs(
        pipe, x_row, num_cols,
        kernel=kernel, beta=beta, sigma=sigma,
        n=n_smooth, batch_size=batch_size,
        seed=seed + 1,
    )

    # craft adversarial example in *tabular space*
    x_adv_row = attack_hybrid_tabular_alternating(
        pipe,
        x_row,
        num_cols=num_cols,
        cat_cols=cat_cols,
        eps_l2=attack_eps_l2,
        steps=attack_steps,
        step=attack_step,
        d_cat=attack_d_cat,
        seed=seed + 2,
        log=False,
        y_true=y_true,
    )

    # base predictions on clean/adv (still evaluated via pipeline)
    y_base_clean = int(res_clean["y_base"])
    y_base_adv = int(pipe.predict(x_adv_row.to_frame().T)[0])
    
    
    
    

    # smoothed prediction on adversarial row (center smoothing at x_adv_row)
    res_adv = predict_hybrid_rs(
        pipe, x_adv_row, num_cols,
        kernel=kernel, beta=beta, sigma=sigma,
        n=n_smooth, batch_size=batch_size,
        seed=seed + 3,
    )

    return {
        "y_true": int(y_true),
        "clean": {
            "y_base": y_base_clean,
            "y_smooth": int(res_clean["y_smooth"]),
            "p_hat": float(res_clean["p_hat"]),
            "counts": res_clean["counts"],
            "correct_base": bool(y_base_clean == int(y_true)),
            "correct_smooth": bool(int(res_clean["y_smooth"]) == int(y_true)),
        },
        "adv": {
            "y_base": y_base_adv,
            "y_smooth": int(res_adv["y_smooth"]),
            "p_hat": float(res_adv["p_hat"]),
            "counts": res_adv["counts"],
            "correct_base": bool(y_base_adv == int(y_true)),
            "correct_smooth": bool(int(res_adv["y_smooth"]) == int(y_true)),
        },
        "attack": {
            "eps_l2": float(attack_eps_l2),
            "d_cat": int(attack_d_cat),
            "steps": int(attack_steps),
            "step": float(attack_step),
        },
        "smoothing": {
            "kernel": kernel,
            "beta": float(beta),
            "sigma": float(sigma),
            "n": int(n_smooth),
        },
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--C", type=float, default=1.0)

    # certification params
    ap.add_argument("--kernel", choices=["absorb", "uniform"], default="uniform")
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--r_max", type=float, default=10.0)
    ap.add_argument("--d_max_report", type=int, default=2)

    # empirical attack params
    ap.add_argument("--attack_eps_l2", type=float, default=1.0)
    ap.add_argument("--attack_steps", type=int, default=30)
    ap.add_argument("--attack_step", type=float, default=0.2)
    ap.add_argument("--attack_d_cat", type=int, default=2)
    ap.add_argument("--n_attack_eval", type=int, default=500)
    
    
    # data augmentation params
    ap.add_argument("--K_aug", type=int, default=5)
    ap.add_argument("--sigma_train", type=float, default=0.5)
    ap.add_argument("--beta_train", type=float, default=0.25)

    # how many examples to certify
    ap.add_argument("--n_certify", type=int, default=30)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------
    # Train model
    # ------------------------
    Xtr, Xte, ytr, yte, num_cols, cat_cols = load_adult(seed=args.seed)
    
    # 1) fit preprocessor on clean training data
    pipe = build_pipeline(num_cols, cat_cols, C=args.C)
    pipe.named_steps["pre"].fit(Xtr)

    # 2) augment in raw space using that fixed pre
    Xtr_aug, ytr_aug = augment_train_tabular(
        pipe, Xtr, ytr,
        num_cols=num_cols, cat_cols=cat_cols,
        K=args.K_aug,
        sigma=args.sigma_train,
        beta=args.beta_train,
        seed=args.seed,
    )

    # 3) train ONLY the SVM on transformed augmented data (do not refit pre)
    X_aug_dense = pipe.named_steps["pre"].transform(Xtr_aug)
    svm = clone(pipe.named_steps["svm"])
    svm.fit(X_aug_dense, ytr_aug)

    # 4) reassemble fitted pipeline
    pipe = Pipeline([("pre", pipe.named_steps["pre"]), ("svm", svm)])



    # pipe = build_pipeline(num_cols, cat_cols, C=args.C)
    # pipe.fit(Xtr, ytr)

    ypred = pipe.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print(f"Adult LinearSVC test accuracy: {acc:.4f}")
    
    
    # ------------------------------------------------------------
    # Sanity check: Hybrid RS prediction under tabular attack
    # ------------------------------------------------------------
   
    print("\n[Hybrid RS under attack – sanity check]")

    n_eval = 1000
    idxs = np.random.choice(len(Xte), size=n_eval, replace=False)

    stats = {
        "base_clean": 0,
        "base_adv": 0,
        "smooth_clean": 0,
        "smooth_adv": 0,
    }

    for k, idx in enumerate(idxs):
        x_row = Xte.iloc[idx]
        y_true = int(yte[idx])
        
        res_certify = certify_example(
            pipe, 
            x_row, 
            num_cols,
            kernel=args.kernel,
            beta=float(args.beta),
            sigma=float(args.sigma),
            tau=float(args.tau),
            alpha=float(args.alpha),
            n=int(args.n),
            batch_size=int(args.batch_size),
            r_max=float(args.r_max),
            seed=args.seed + 200000 + k,
            d_max_report=args.d_max_report,
        )

        res = eval_under_attack_with_smoothed(
            pipe,
            x_row,
            y_true,
            num_cols=num_cols,
            cat_cols=cat_cols,
            attack_eps_l2=args.attack_eps_l2,
            attack_steps=args.attack_steps,
            attack_step=args.attack_step,
            attack_d_cat=args.attack_d_cat,
            kernel=args.kernel,
            beta=args.beta,
            sigma=args.sigma,
            n_smooth=args.n,
            batch_size=args.batch_size,
            seed=args.seed + 50000 + k,
        )

        stats["base_clean"] += res["clean"]["correct_base"]
        stats["smooth_clean"] += res["clean"]["correct_smooth"]
        stats["base_adv"] += res["adv"]["correct_base"]
        stats["smooth_adv"] += res["adv"]["correct_smooth"]
        
        print("\nCertification results for this example:", res_certify["certified"])
        print(
            f"[ex {k}] "
            f"y={y_true} | "
            f"base clean={res['clean']['y_base']} "
            f"smooth clean={res['clean']['y_smooth']} "
            f"(p̂={res['clean']['p_hat']:.3f}) || "
            f"base adv={res['adv']['y_base']} "
            f"smooth adv={res['adv']['y_smooth']} "
            f"(p̂={res['adv']['p_hat']:.3f})"
        )

    print("\n[Summary]")
    print(f"Base clean accuracy:     {stats['base_clean'] / n_eval:.3f}")
    print(f"Smoothed clean accuracy: {stats['smooth_clean'] / n_eval:.3f}")
    print(f"Base adv accuracy:       {stats['base_adv'] / n_eval:.3f}")
    print(f"Smoothed adv accuracy:   {stats['smooth_adv'] / n_eval:.3f}")
    
    exit()
    # ------------------------
    # Empirical hybrid attack (TABULAR-SPACE, alternating)
    # ------------------------
    idxs = np.random.choice(
        len(Xte),
        size=min(args.n_attack_eval, len(Xte)),
        replace=False,
    )

    ok_base = 0
    ok_stability = 0  # optional: decision stability f(x_adv)=f(x)
    max_examples = 100
    for k, idx in enumerate(idxs):
        x_row = Xte.iloc[idx]
        y_true = int(yte[idx])

        y_clean = int(pipe.predict(x_row.to_frame().T)[0])

        x_adv_row = attack_hybrid_tabular_alternating(
            pipe,
            x_row,
            num_cols=num_cols,
            cat_cols=cat_cols,
            eps_l2=args.attack_eps_l2,
            steps=args.attack_steps,
            step=args.attack_step,
            d_cat=args.attack_d_cat,
            seed=args.seed + 10000 + k,
            steps_per_round=5,   # you can expose as CLI if you want
            log=False,
        )

        y_adv = int(pipe.predict(x_adv_row.to_frame().T)[0])

        if y_adv == y_true:
            ok_base += 1
        if y_adv == y_clean:
            ok_stability += 1
            
        if k < max_examples:
            break

    print(
        f"Empirical robust accuracy (hybrid tabular attack): {ok_base/len(idxs):.4f} "
        f"[eps_l2={args.attack_eps_l2}, d_cat={args.attack_d_cat}, steps={args.attack_steps}]"
    )
    print(
        f"Decision stability under attack: {ok_stability/len(idxs):.4f} "
        f"[fraction with f(x_adv)=f(x)]"
    )

    
  # ------------------------
    # Hybrid RS certification (with stats)
    # ------------------------
    nC = min(args.n_certify, len(Xte))

    # stats containers
    cert_counts = {}     # d -> number of certified points
    cert_r_sum = {}      # d -> sum of certified radii
    any_cert = 0         # number of points with at least one (d,r)>0

    for j in range(nC):
        y_true = int(yte[idx])

        res = certify_example(
            pipe, Xte.iloc[j], num_cols,
            kernel=args.kernel,
            beta=float(args.beta),
            sigma=float(args.sigma),
            tau=float(args.tau),
            alpha=float(args.alpha),
            n=int(args.n),
            batch_size=int(args.batch_size),
            r_max=float(args.r_max),
            seed=args.seed + 200000 + j,
            d_max_report=args.d_max_report,
        )

        cert = res["certified"]
        
        y0 = res["y0"]
        if y0 != y_true:
            is_ok = False
        else:
            is_ok = True

        if len(cert) > 0:
            any_cert += 1

        for (d, r) in cert:
            if d not in cert_counts:
                cert_counts[d] = 0
                cert_r_sum[d] = 0.0
            cert_counts[d] += 1
            cert_r_sum[d] += r

        max_cert = cert[-1] if len(cert) else None
        print(
            f"[cert ex {j}] "
            f"pA={res['pA']:.4f} (nA={res['nA']}/{args.n}) "
            f"V_worst_last={res['V_worst_last']} "
            f"max(d,r)={max_cert}"
            f" | correct={is_ok}"
        )

    # ------------------------
    # Print certification summary
    # ------------------------
    print("\n[Certification summary]")
    print(f"Any certified point: {any_cert}/{nC} = {any_cert/nC:.3f}")

    for d in sorted(cert_counts.keys()):
        frac = cert_counts[d] / nC
        r_avg = cert_r_sum[d] / cert_counts[d]
        print(
            f"d = {d}: "
            f"certified fraction = {frac:.3f}, "
            f"avg certified r = {r_avg:.4f}"
        )


if __name__ == "__main__":
    main()
