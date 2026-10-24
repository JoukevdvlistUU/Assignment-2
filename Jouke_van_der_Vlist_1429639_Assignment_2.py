# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 17:52:35 2026

@author: jouke
"""

# -*- coding: utf-8 -*-
"""
Full reproduction of Hanna et al. (2026), "A systematic comparison of machine learning models for
missing value imputation in household electricity consumption data", Scientific Reports 16:26698.

Follows the paper's methodology:
  Phase 1  all 30 models of Table 6, each scored on all 50 metrics of Appendix 1 (Appendix 2)
  Phase 2  the top three models by MAE combined with Bagging, Boosting and Stacking (Appendix 3)
  Phase 3  three-stage hybrid with error-curve learning on the same top three (Appendix 4)
  plus the eight core metrics (Tables 7 to 10), a cross-phase ranking and actual-vs-predicted
  figures (Figs. 4 to 6). Every number is computed from household_power_consumption.txt.

Runs entirely on the CPU; all models use the library defaults unless stated under Settings.
All results are printed to the console; no files or folders are created.

One file, run top to bottom. In Spyder: press F5, or Ctrl+Enter per cell (# %%).
Put household_power_consumption.txt in the same folder as this file, or set DATA_PATH below.
"""

# %% 1. Imports
import os
import time
import warnings

import holidays
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from joblib import parallel_config
parallel_config(backend="threading")   # threads, not processes: Bagging otherwise fails in Spyder ("failed to un-serialize")
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import (BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression,
                                  OrthogonalMatchingPursuit, PassiveAggressiveRegressor,
                                  RANSACRegressor, Ridge, TweedieRegressor)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")              # convergence and deprecation notices of default settings

try:  # work in the folder of this script so the data file is found
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

# %% 2. Settings
DATA_PATH = "household_power_consumption.txt"
CHUNK_ROWS = 200_000          # rows read per chunk; lower this if reading runs out of memory

FEATURES = "paper+lag1min"    # "paper" | "paper+lag1min" | "paper+electrical" | "gap-realistic"
SAMPLE = 100_000              # rows drawn at random from the usable minutes; 0 = all rows (paper; very slow)
SPLIT = "random"              # "random" 80/20 as in the paper | "chronological" (last 20% as test)
TEST_SIZE = 0.2
USE_GA = True                 # Genetic Algorithm feature selection
SCALE_FEATURES = True         # "Data normalization ensured consistent feature scales" (StandardScaler)
CV = 3                        # folds for out-of-fold predictions (stacking, boosting, hybrid)
N_TOP = 3                     # number of best phase-1 models (by MAE) used in phases 2 and 3
SEED = 42
N_JOBS = -1                   # -1 = all CPU cores; set 1 if Spyder has trouble with parallel jobs

# GA settings (not reported in the paper; our choice)
GA_POP, GA_GENERATIONS, GA_MUTATION, GA_ELITE = 16, 8, 0.05, 2
GA_FIT_ROWS, GA_VAL_ROWS, GA_TREES = 30_000, 10_000, 20
GA_INIT_P, GA_TOURNAMENT, GA_CROSSOVER_P = 0.7, 3, 0.5

# Model settings the paper leaves open (all other models use library defaults)
POLY_DEGREE = 2               # Polynomial Regression
PCR_COMPONENTS = 2            # Principal Component Regression (same as the PLS default)
SVM_MAX_ROWS = 20_000         # kernel SVR scales quadratically; trained on at most this many rows
NN_EPOCHS, NN_BATCH, NN_LR, NN_PATIENCE, NN_VAL_FRAC, NN_HIDDEN = 30, 256, 1e-3, 5, 0.1, 64

# Metric settings (Appendix 1 leaves these open)
BENCHMARK = "naive_1min"      # benchmark forecast for relative metrics: "naive_1min" | "naive_24h" | "zero" | "train_mean"
CRITICAL_MARGIN = 0.10        # "critical event": |error| > margin x |actual|
TWEEDIE_POWER = 0             # power of the mean Tweedie deviance (0 gives MSE, as in the paper's MTD column)
EPS = 1e-9                    # guards against log(0) and division by zero

# Feature settings (minutes)
DAY_START, DAY_END = 7, 19
LAG_MINUTES = {"lag_1h": 60, "lag_24h": 1440, "lag_168h": 10080, "lag_720h": 43200}
ROLL_MINUTES = {"24h": 1440, "168h": 10080}
DIFF_MINUTES = {"diff_1h": 60, "diff_24h": 1440, "diff_168h": 10080}

# Category boundaries used to bin the gaps (definitions of the paper's Table 2, not results)
GAP_BINS = [("<=10 min", 1, 10), ("11-60 min", 11, 60), ("1-6 h", 61, 360),
            ("6-24 h", 361, 1440), (">24 h", 1441, None)]

MAKE_FIGURES = True           # show actual vs predicted plots (Spyder Plots pane; nothing is saved)
FIG_POINTS = 50

T_START = time.time()
TORCH_DEVICE = "cpu"
XGB_DEVICE = "cpu"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log(f"PyTorch {torch.__version__} on {TORCH_DEVICE}; XGBoost on {XGB_DEVICE}")

# %% 3. Load data (in chunks so the parser never needs much memory at once)
NUM_COLS = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
TARGET = "Global_active_power"

chunks = []
for ch in pd.read_csv(DATA_PATH, sep=";", na_values="?", chunksize=CHUNK_ROWS,
                      dtype={**{c: "float32" for c in NUM_COLS}, "Date": str, "Time": str}):
    ch.index = pd.to_datetime(ch["Date"] + " " + ch["Time"], format="%d/%m/%Y %H:%M:%S")
    chunks.append(ch.drop(columns=["Date", "Time"]))
raw = pd.concat(chunks)
del chunks
log(f"loaded {len(raw):,} rows from {raw.index.min()} to {raw.index.max()}")

# %% 4. Missing data structure, computed from the txt file (Tables 1, 2 and 3 of the paper)
na = raw[NUM_COLS].isna()
n_rows = len(raw)
n_missing = int(na[TARGET].sum())
log(f"missing values per column: {na.sum().to_dict()}")
log(f"rows with any missing value: {int(na.any(axis=1).sum()):,}, rows with all values missing: "
    f"{int(na.all(axis=1).sum()):,} ({100 * n_missing / n_rows:.2f}% of rows)")

miss = na[TARGET].to_numpy().astype(int)
d = np.diff(np.r_[0, miss, 0])
gap_start, gap_end = np.where(d == 1)[0], np.where(d == -1)[0]
gap_len = gap_end - gap_start
log(f"contiguous gaps: {len(gap_len)}")
table2 = []
for name, lo, hi in GAP_BINS:
    sel = (gap_len >= lo) & (gap_len <= (hi if hi is not None else gap_len.max()))
    table2.append({"category": name, "gaps": int(sel.sum()), "total_min": int(gap_len[sel].sum()),
                   "mean_min": round(float(gap_len[sel].mean()), 1) if sel.any() else 0,
                   "max_min": int(gap_len[sel].max()) if sel.any() else 0,
                   "pct_missing": round(100 * gap_len[sel].sum() / gap_len.sum(), 2)})
table2 = pd.DataFrame(table2)
print("\nTable 2, gap duration distribution:\n", table2.to_string(index=False))
long_gaps = gap_len > GAP_BINS[-1][1] - 1
table3 = pd.DataFrame({"start": raw.index[gap_start[long_gaps]], "end": raw.index[gap_end[long_gaps] - 1],
                       "minutes": gap_len[long_gaps], "days": np.round(gap_len[long_gaps] / 1440, 2)})
print("\nTable 3, long gaps:\n", table3.to_string(index=False))
yearly = (na[TARGET].groupby(raw.index.year).mean() * 100).round(2)
print("\nMissing rate per year (%):\n", yearly.to_string())
table1 = raw[[TARGET, "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].astype("float64").agg(
    ["mean", "min", "max", "std", "skew", "kurt"]).T
print("\nTable 1, summary statistics:\n", table1.round(3).to_string())

# %% 5. Baseline: time-specific mean imputation
# Mean of the same month, day, hour and minute in the other years; minute-of-day mean as fallback.
df = raw.copy()
key_date = [raw.index.month, raw.index.day, raw.index.hour, raw.index.minute]
key_tod = [raw.index.hour, raw.index.minute]
for c in NUM_COLS:
    s = raw[c]
    df[c] = s.fillna(s.groupby(key_date).transform("mean")).fillna(s.groupby(key_tod).transform("mean"))
log("baseline imputation done")

# %% 6. Feature engineering
idx = df.index
g = df[TARGET].astype("float32")
F = pd.DataFrame(index=idx)
F["year"] = idx.year; F["month"] = idx.month; F["week"] = idx.isocalendar().week.to_numpy()
F["day"] = idx.day; F["dayofyear"] = idx.dayofyear; F["dayofweek"] = idx.dayofweek
F["hour"] = idx.hour; F["minute"] = idx.minute
F["is_day"] = ((idx.hour >= DAY_START) & (idx.hour < DAY_END)).astype("int8")
F["is_weekend"] = (idx.dayofweek >= 5).astype("int8")
fr_holidays = holidays.France(years=range(idx.year.min(), idx.year.max() + 1))
F["is_holiday"] = pd.Index(idx.date).isin(list(fr_holidays.keys())).astype("int8")
for c in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
    F[c] = df[c].astype("float32")
for name, k in LAG_MINUTES.items():
    F[name] = g.shift(k)
past = g.shift(1)
for name, w in ROLL_MINUTES.items():
    r = past.rolling(w, min_periods=w)
    F[f"roll_mean_{name}"] = r.mean(); F[f"roll_std_{name}"] = r.std()
    F[f"roll_min_{name}"] = r.min(); F[f"roll_max_{name}"] = r.max()
for name, k in DIFF_MINUTES.items():
    F[name] = g.shift(k) - g.shift(2 * k)
F["lag_1min"] = g.shift(1)
F["lag_1day_same_minute"] = g.shift(1440)           # used only as benchmark forecast
for c in ["Global_intensity", "Voltage", "Global_reactive_power"]:
    F[c] = df[c].astype("float32")
F = F.astype("float32")
F["y"] = g
F = F.dropna()
del raw, df, past
log(f"features built: {F.shape[0]:,} usable rows (first {n_rows - F.shape[0]:,} minutes dropped for the longest lag)")

CALENDAR = ["year", "month", "week", "day", "dayofyear", "dayofweek", "hour", "minute",
            "is_day", "is_weekend", "is_holiday"]
SUBMETERS = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
HISTORY = (list(LAG_MINUTES) + [f"roll_{s}_{w}" for w in ROLL_MINUTES for s in ["mean", "std", "min", "max"]]
           + list(DIFF_MINUTES))
FEATURE_SETS = {
    "paper": CALENDAR + SUBMETERS + HISTORY,
    "paper+lag1min": CALENDAR + SUBMETERS + HISTORY + ["lag_1min"],
    "paper+electrical": CALENDAR + SUBMETERS + HISTORY + ["Global_intensity", "Voltage", "Global_reactive_power"],
    "gap-realistic": CALENDAR + [n for n, k in LAG_MINUTES.items() if k >= 10080],
}
y_all = F["y"].to_numpy(dtype="float64")


# %% 7. Helper functions: split, the 50 metrics of Appendix 1, ranking, wrappers
def sample_and_split(n_sample, split, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = np.arange(len(F)) if n_sample == 0 else np.sort(rng.choice(len(F), n_sample, replace=False))
    if split == "random":
        perm = rng.permutation(rows)
        n_te = int(round(TEST_SIZE * len(rows)))
        te, tr = np.sort(perm[:n_te]), np.sort(perm[n_te:])
        series = y_all
    else:
        cut = int(round((1 - TEST_SIZE) * len(rows)))
        tr, te = rows[:cut], rows[cut:]
        series = y_all[:rows[cut]]
    # in-sample one-step naive errors of the training period, used by MASE, MdASE and RMSSE
    dy = np.diff(series)
    return tr, te, np.mean(np.abs(dy)), np.mean(dy ** 2)


def all_metrics(y, p, yb, groups, scale_abs, scale_sq):
    """The 50 metrics of Appendix 1. y actual, p prediction, yb benchmark forecast,
    groups = series labels for AvgRelMAE (test-set years)."""
    p = np.asarray(p, dtype="float64").ravel()
    e, eb = y - p, y - yb
    ae, aeb = np.abs(e), np.abs(eb)
    pe = e / y                                          # target is strictly positive in this dataset
    rng_y = y.max() - y.min()
    w = y                                               # weights: consumption level (emphasises peaks)
    ok = aeb > 0                                        # relative errors need a nonzero benchmark error
    rel = ae[ok] / aeb[ok]
    lp = np.log1p(np.clip(p, 0, None)); ly = np.log1p(y)
    grmse = lambda err: np.exp(np.mean(np.log(err ** 2 + EPS)) / 2)
    rel_by_group = [np.mean(ae[groups == gname]) / np.mean(aeb[groups == gname])
                    for gname in np.unique(groups) if np.mean(aeb[groups == gname]) > 0]
    if TWEEDIE_POWER == 0:
        mtd = np.mean(e ** 2)
    else:
        from sklearn.metrics import mean_tweedie_deviance
        mtd = mean_tweedie_deviance(y, np.clip(p, EPS, None), power=TWEEDIE_POWER)
    m = {
        # scale-dependent
        "MSE": np.mean(e ** 2), "RMSE": np.sqrt(np.mean(e ** 2)), "MAE": ae.mean(), "MdAE": np.median(ae),
        "RMdSE": np.sqrt(np.median(e ** 2)), "ME": e.mean(),
        # percentage
        "MAPE": 100 * np.mean(np.abs(pe)), "MdAPE": 100 * np.median(np.abs(pe)),
        "RMSPE": 100 * np.sqrt(np.mean(pe ** 2)), "RMdSPE": 100 * np.sqrt(np.median(pe ** 2)),
        "sMAPE": 100 * np.mean(2 * ae / (np.abs(y) + np.abs(p) + EPS)),
        "sMdAPE": 100 * np.median(2 * ae / (np.abs(y) + np.abs(p) + EPS)),
        "msMAPE": 200 * np.mean(ae / np.where(np.abs(y + p) < EPS, EPS, y + p)),
        "MAAPE": np.mean(np.arctan(np.abs(pe))), "MPE": 100 * np.mean(pe),
        # relative
        "MRAE": rel.mean(), "MdRAE": np.median(rel), "RMRSE": np.sqrt(np.mean(rel ** 2)),
        "GMRAE": np.exp(np.mean(np.log(rel + EPS))), "RGRMSE": grmse(e) / grmse(eb),
        "RelMAE": ae.mean() / aeb.mean(), "RelMSE": np.mean(e ** 2) / np.mean(eb ** 2),
        "RelRMSE": np.sqrt(np.mean(e ** 2) / np.mean(eb ** 2)),
        "RRSE": np.sqrt(np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
        "AvgRelMAE": np.exp(np.mean(np.log(rel_by_group))),
        # scaled
        "MASE": ae.mean() / scale_abs, "MdASE": np.median(ae) / scale_abs,
        "RMSSE": np.sqrt(np.mean(e ** 2) / scale_sq), "sME": e.mean() / rng_y,
        "sMSE": np.mean(e ** 2) / rng_y ** 2, "sMdAE": np.median(ae) / rng_y,
        # geometric
        "GRMSE": grmse(e), "GMAE": np.exp(np.mean(np.log(ae + EPS))),
        # logarithmic
        "RMSLE": np.sqrt(np.mean((ly - lp) ** 2)), "NWRMSLE": np.sqrt(np.sum(w * (ly - lp) ** 2) / np.sum(w)),
        # rate-based
        "MSR": np.mean((e / np.where(np.abs(y + p) < EPS, EPS, y + p)) ** 2),
        "MAR": 100 * np.mean(ae / np.where(np.abs(y + p) < EPS, EPS, y + p)),
        # weighted
        "WMAE": np.sum(ae * w) / np.sum(w), "WAPE": 100 * ae.sum() / y.sum(),
        "sWAPE": 100 * np.sum(2 * ae) / np.sum(np.abs(y) + np.abs(p)),
        "WRMSPE": 100 * np.sqrt(np.sum(w * pe ** 2) / np.sum(w)),
        # normalized
        "ND": ae.sum() / np.abs(y).sum(), "NRMSE": np.sqrt(np.mean(e ** 2)) / rng_y,
        # correlation-based
        "CORR": np.corrcoef(y, p)[0, 1], "R2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
        # rank/counting-based
        "PB_score": 100 * np.mean(ae < aeb), "Critical_event": 100 * np.mean(ae > CRITICAL_MARGIN * np.abs(y)),
        # specialized
        "RTAE": ae.sum() / np.abs(y - y.mean()).sum(), "EVS": 1 - np.var(e) / np.var(y), "MTD": mtd,
    }
    return m


CORE = ["MAE", "RMSE", "MAPE", "R2", "MdAE", "ME", "MASE", "CORR"]
HIGHER_BETTER = {"CORR", "R2", "EVS", "PB_score"}
ZERO_BETTER = {"ME", "MPE", "sME"}


def average_rank(table):
    """Rank every model per metric (1 = best) and average over all metrics."""
    ranks = pd.DataFrame(index=table.index)
    for col in table.columns:
        v = table[col].astype(float)
        if col in HIGHER_BETTER:
            ranks[col] = (-v).rank(method="min")
        elif col in ZERO_BETTER:
            ranks[col] = v.abs().rank(method="min")
        else:
            ranks[col] = v.rank(method="min")
    return ranks.mean(axis=1).sort_values()


def oof_predict(model, X, y, folds):
    """Out-of-fold predictions: each training row is predicted by a model that never saw it."""
    p = np.empty(len(y), dtype="float64")
    for tr_i, va_i in folds:
        p[va_i] = np.asarray(clone(model).fit(X[tr_i], y[tr_i]).predict(X[va_i])).ravel()
    return p


class SubsampledRegressor(BaseEstimator, RegressorMixin):
    """Trains the wrapped model on at most max_rows random training rows (used for kernel SVR)."""
    def __init__(self, estimator=None, max_rows=20_000, random_state=42):
        self.estimator, self.max_rows, self.random_state = estimator, max_rows, random_state

    def fit(self, X, y):
        i = np.arange(len(y))
        if len(y) > self.max_rows:
            i = np.random.default_rng(self.random_state).choice(len(y), self.max_rows, replace=False)
        self.model_ = clone(self.estimator).fit(X[i], y[i])
        return self

    def predict(self, X):
        return self.model_.predict(X)


class TorchRegressor(BaseEstimator, RegressorMixin):
    """DNN, LSTM, CNN and RNN in PyTorch. The feature vector is fed as a sequence of length
    n_features with one channel to the LSTM, RNN and CNN (the paper does not describe the input)."""
    def __init__(self, arch="dnn", hidden=64, epochs=30, batch_size=256, lr=1e-3, patience=5,
                 val_frac=0.1, device="cpu", random_state=42):
        self.arch, self.hidden, self.epochs, self.batch_size, self.lr = arch, hidden, epochs, batch_size, lr
        self.patience, self.val_frac, self.device, self.random_state = patience, val_frac, device, random_state

    def _build(self, n_in):
        nn, h = torch.nn, self.hidden
        if self.arch == "dnn":
            return nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        class Seq(nn.Module):
            def __init__(self, cell):
                super().__init__()
                self.rnn = (nn.LSTM if cell == "lstm" else nn.RNN)(1, h, batch_first=True)
                self.out = nn.Linear(h, 1)

            def forward(self, x):
                o, _ = self.rnn(x.unsqueeze(-1))
                return self.out(o[:, -1, :])

        class CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
                                          nn.Conv1d(32, 64, 3, padding=1), nn.ReLU())
                self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * n_in, h), nn.ReLU(), nn.Linear(h, 1))

            def forward(self, x):
                return self.head(self.conv(x.unsqueeze(1)))

        return CNN() if self.arch == "cnn" else Seq(self.arch)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype="float32"); y = np.asarray(y, dtype="float32")
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std()) or 1.0
        Xt = torch.from_numpy(X).to(self.device)
        yt = torch.from_numpy((y - self.y_mean_) / self.y_std_).to(self.device).unsqueeze(1)
        perm = rng.permutation(len(X)); n_val = max(1, int(self.val_frac * len(X)))
        va, trn = perm[:n_val], perm[n_val:]
        model = self._build(X.shape[1]).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.epochs):
            model.train()
            order = trn[rng.permutation(len(trn))]
            for b in range(0, len(order), self.batch_size):
                bi = torch.from_numpy(order[b:b + self.batch_size]).to(self.device)
                opt.zero_grad()
                loss = loss_fn(model(Xt[bi]), yt[bi])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vi = torch.from_numpy(va).to(self.device)
                v = loss_fn(model(Xt[vi]), yt[vi]).item()
            if v < best:
                best, wait = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        model.load_state_dict(best_state)
        self.model_ = model
        return self

    def predict(self, X):
        X = torch.from_numpy(np.asarray(X, dtype="float32")).to(self.device)
        self.model_.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(X), 8192):
                out.append(self.model_(X[b:b + 8192]).cpu().numpy().ravel())
        return np.concatenate(out) * self.y_std_ + self.y_mean_


def make_models():
    """The 30 models of Table 6, with library defaults unless stated in the settings."""
    nn_args = dict(hidden=NN_HIDDEN, epochs=NN_EPOCHS, batch_size=NN_BATCH, lr=NN_LR, patience=NN_PATIENCE,
                   val_frac=NN_VAL_FRAC, device=TORCH_DEVICE, random_state=SEED)
    return {
        # linear
        "LR": LinearRegression(), "Ridge": Ridge(), "LASSO": Lasso(), "ElasticNet": ElasticNet(),
        "BayesianRidge": BayesianRidge(),
        "Poly": make_pipeline(PolynomialFeatures(POLY_DEGREE, include_bias=False), LinearRegression()),
        "TRM": TweedieRegressor(), "PARM": PassiveAggressiveRegressor(random_state=SEED),
        # tree-based
        "DT": DecisionTreeRegressor(random_state=SEED),
        "RF": RandomForestRegressor(random_state=SEED, n_jobs=N_JOBS),
        "GB": GradientBoostingRegressor(random_state=SEED),
        "XGBoost": XGBRegressor(random_state=SEED, n_jobs=N_JOBS, tree_method="hist", device=XGB_DEVICE),
        "CatBoost": CatBoostRegressor(random_seed=SEED, verbose=0, thread_count=-1, allow_writing_files=False),
        "AdaBoost": AdaBoostRegressor(random_state=SEED),
        "Bagging": BaggingRegressor(random_state=SEED, n_jobs=N_JOBS),
        "ETR": ExtraTreesRegressor(random_state=SEED, n_jobs=N_JOBS),
        "HistGBR": HistGradientBoostingRegressor(random_state=SEED),
        "LGBM": LGBMRegressor(random_state=SEED, n_jobs=N_JOBS, verbose=-1),
        # other
        "SVM": SubsampledRegressor(SVR(), SVM_MAX_ROWS, SEED),
        "KNN": KNeighborsRegressor(n_jobs=N_JOBS),
        "Huber": HuberRegressor(), "RANSAC": RANSACRegressor(random_state=SEED),
        "PLS": PLSRegression(), "OMP": OrthogonalMatchingPursuit(),
        "PCR": make_pipeline(PCA(n_components=PCR_COMPONENTS, random_state=SEED), LinearRegression()),
        # neural networks
        "MLP": MLPRegressor(random_state=SEED),
        "DNN": TorchRegressor("dnn", **nn_args), "LSTM": TorchRegressor("lstm", **nn_args),
        "CNN": TorchRegressor("cnn", **nn_args), "RNN": TorchRegressor("rnn", **nn_args),
    }


def plot_actual_vs_predicted(preds, y, title):
    if not MAKE_FIGURES:
        return
    names = list(preds)
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3 * nrow), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        ax.plot(y[:FIG_POINTS], label="Actual")
        ax.plot(np.asarray(preds[name]).ravel()[:FIG_POINTS], "--", label="Predicted")
        ax.set_title(name, fontsize=9); ax.set_xlabel("Time Length"); ax.set_ylabel("Active Power (kW)")
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(title); fig.tight_layout(rect=(0, 0, 1, 0.98))
    plt.show()


# %% 8. Sample, split, benchmark and scaling
cols = FEATURE_SETS[FEATURES]
tr, te, scale_abs, scale_sq = sample_and_split(SAMPLE, SPLIT)
X = F[cols].to_numpy(dtype="float64")
Xtr, Xte, ytr, yte = X[tr], X[te], y_all[tr], y_all[te]
groups_te = F["year"].to_numpy()[te]
yb_te = {"naive_1min": F["lag_1min"].to_numpy(dtype="float64")[te],
         "naive_24h": F["lag_1day_same_minute"].to_numpy(dtype="float64")[te],
         "zero": np.zeros(len(te)),
         "train_mean": np.full(len(te), ytr.mean())}[BENCHMARK]
log(f"features={FEATURES} ({len(cols)}), split={SPLIT}, train={len(tr):,}, test={len(te):,}, benchmark={BENCHMARK}")


def score(preds):
    return pd.DataFrame({m: all_metrics(yte, p, yb_te, groups_te, scale_abs, scale_sq)
                         for m, p in preds.items()}).T


# %% 9. Genetic Algorithm feature selection
if USE_GA:
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(Xtr))
    n_fit = min(GA_FIT_ROWS, int(0.75 * len(Xtr)))
    fit_i, val_i = perm[:n_fit], perm[n_fit:n_fit + GA_VAL_ROWS]
    cache = {}

    def fitness(mask):
        key = mask.tobytes()
        if key not in cache:
            if mask.sum() == 0:
                cache[key] = np.inf
            else:
                m = ExtraTreesRegressor(n_estimators=GA_TREES, random_state=SEED, n_jobs=N_JOBS)
                m.fit(Xtr[fit_i][:, mask], ytr[fit_i])
                cache[key] = mean_absolute_error(ytr[val_i], m.predict(Xtr[val_i][:, mask]))
        return cache[key]

    n_feat = len(cols)
    pop = rng.random((GA_POP, n_feat)) < GA_INIT_P
    pop[0] = True
    for gen in range(GA_GENERATIONS):
        scores = np.array([fitness(ind) for ind in pop])
        order = np.argsort(scores)
        log(f"GA generation {gen}: best validation MAE {scores[order[0]]:.4f} with {pop[order[0]].sum()} features")
        new = [pop[i].copy() for i in order[:GA_ELITE]]
        while len(new) < GA_POP:
            a, b = (min(rng.choice(GA_POP, GA_TOURNAMENT, replace=False), key=lambda i: scores[i]) for _ in range(2))
            child = np.where(rng.random(n_feat) < GA_CROSSOVER_P, pop[a], pop[b])
            child ^= rng.random(n_feat) < GA_MUTATION
            new.append(child)
        pop = np.array(new)
    best = pop[np.argmin([fitness(ind) for ind in pop])]
    cols = [c for c, keep in zip(cols, best) if keep]
    Xtr, Xte = Xtr[:, best], Xte[:, best]
    log(f"GA kept {len(cols)} features: {cols}")

if SCALE_FEATURES:
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

# %% 10. Phase 1: all 30 individual models (Tables 7 and 8, Appendix 2, Fig. 4)
models = make_models()
preds1, fit_time = {}, {}
for name, model in models.items():
    t0 = time.time()
    try:
        preds1[name] = np.asarray(clone(model).fit(Xtr, ytr).predict(Xte), dtype="float64").ravel()
        fit_time[name] = time.time() - t0
        log(f"phase 1: {name:<14} MAE {np.mean(np.abs(yte - preds1[name])):.4f}  ({fit_time[name]:.0f} s)")
    except Exception as ex:                          # keep going if one model fails
        log(f"phase 1: {name} FAILED: {ex}")
missing = [n for n in models if n not in preds1]
if missing:
    raise SystemExit(f"Phase 1 incomplete, failed: {missing}. The top 3 for phases 2 and 3 needs all 30 models.")
appendix2 = score(preds1)
appendix2["fit_seconds"] = pd.Series(fit_time)
table7_8 = appendix2[CORE].sort_values("MAE")
print("\nPhase 1, eight core metrics (sorted by MAE):\n", table7_8.round(4).to_string())
bench = score({f"Benchmark ({BENCHMARK})": yb_te})
print("\nBenchmark forecast, eight core metrics:\n", bench[CORE].round(4).to_string())
plot_actual_vs_predicted(preds1, yte, "Phase 1: actual vs predicted")

# %% 11. Phase 2: ensembles of the top models by MAE (Table 9, Appendix 3, Fig. 5)
top = table7_8.index[:N_TOP].tolist()
log(f"top {N_TOP} models by MAE: {top}")
folds = list(KFold(CV, shuffle=True, random_state=SEED).split(Xtr))
oof = {}
for name in top:
    log(f"phase 2: out-of-fold predictions {name}")
    oof[name] = oof_predict(models[name], Xtr, ytr, folds)

preds2 = {}
# Bagging: each top model trained on its own bootstrap sample, predictions averaged
rng = np.random.default_rng(SEED)
bag_preds = []
for name in top:
    log(f"phase 2: ensemble bagging, {name} on a bootstrap sample")
    b = rng.integers(0, len(ytr), len(ytr))
    bag_preds.append(np.asarray(clone(models[name]).fit(Xtr[b], ytr[b]).predict(Xte)).ravel())
preds2["Bagging"] = np.mean(bag_preds, axis=0)
# Boosting: each next model fits the remaining (out-of-fold) residuals of the previous ones
log("phase 2: ensemble boosting")
resid = ytr - oof[top[0]]
boost = preds1[top[0]].copy()
for k, name in enumerate(top[1:]):
    boost += np.asarray(clone(models[name]).fit(Xtr, resid).predict(Xte)).ravel()
    if k < len(top) - 2:
        resid = resid - oof_predict(models[name], Xtr, resid, folds)
preds2["Boosting"] = boost
# Stacking: linear meta-model on out-of-fold predictions
meta = LinearRegression().fit(np.column_stack([oof[n] for n in top]), ytr)
preds2["Stacking"] = meta.predict(np.column_stack([preds1[n] for n in top]))
log(f"phase 2: stacking weights {dict(zip(top, np.round(meta.coef_, 3)))}, intercept {meta.intercept_:.4f}")
appendix3 = score(preds2)
print("\nPhase 2, ensembles, eight core metrics:\n", appendix3[CORE].round(4).to_string())
plot_actual_vs_predicted(preds2, yte, f"Phase 2: ensembles of {', '.join(top)}")

# %% 12. Phase 3: three-stage hybrid with error-curve learning (Table 10, Appendix 4, Fig. 6)
# M1: initial prediction and error curve (out-of-fold, so the error curve is not trivially zero)
# M2: predicts the error curve from the features and the M1 prediction
# M3: final prediction from the features plus the predicted error
m1, m2, m3 = top[0], top[1 % N_TOP], top[2 % N_TOP]
log(f"phase 3: M1 = {m1}, M2 = {m2}, M3 = {m3}")
p1_tr, p1_te = oof[m1], preds1[m1]
err = ytr - p1_tr
X2tr, X2te = np.column_stack([Xtr, p1_tr]), np.column_stack([Xte, p1_te])
err_hat_tr = oof_predict(models[m2], X2tr, err, folds)
err_hat_te = np.asarray(clone(models[m2]).fit(X2tr, err).predict(X2te)).ravel()
X3tr, X3te = np.column_stack([Xtr, err_hat_tr]), np.column_stack([Xte, err_hat_te])
preds3 = {"3-stage hybrid model": np.asarray(clone(models[m3]).fit(X3tr, ytr).predict(X3te)).ravel()}
appendix4 = score(preds3)
print("\nPhase 3, hybrid, eight core metrics:\n", appendix4[CORE].round(4).to_string())
plot_actual_vs_predicted(preds3, yte, f"Phase 3: three-stage hybrid ({m1}, {m2}, {m3})")

# %% 13. Cross-phase summary, all 50 metrics and the settings of this run
allres = pd.concat([appendix2.drop(columns="fit_seconds"),
                    appendix3.rename(index=lambda n: f"Ensemble {n}"),
                    appendix4])
summary = allres[CORE].copy()
summary["avg_rank_50_metrics"] = average_rank(allres)
summary = summary.sort_values("MAE")
print("\nCross-phase summary (eight core metrics and average rank over all 50 metrics):\n",
      summary.round(4).to_string())
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 100)
print("\nAppendix 2, all 50 metrics per model:\n", appendix2.round(4).T.to_string())
print("\nAppendix 3, ensembles, all 50 metrics:\n", appendix3.round(4).T.to_string())
print("\nAppendix 4, hybrid, all 50 metrics:\n", appendix4.round(4).T.to_string())
print("\nSettings of this run:")
for k, v in {"features": FEATURES, "features_used": cols, "sample": SAMPLE, "split": SPLIT,
             "test_size": TEST_SIZE, "seed": SEED, "cv": CV, "ga": USE_GA, "scaled": SCALE_FEATURES,
             "top_models": top, "hybrid": [m1, m2, m3], "benchmark": BENCHMARK, "svm_max_rows": SVM_MAX_ROWS,
             "critical_margin": CRITICAL_MARGIN, "tweedie_power": TWEEDIE_POWER,
             "n_train": int(len(tr)), "n_test": int(len(te)), "mase_scale": round(float(scale_abs), 4)}.items():
    print(f"  {k:<16} {v}")
log(f"finished in {(time.time() - T_START) / 60:.1f} minutes")