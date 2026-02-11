# %% Libraries
import math
import ast
import numpy as np
import pandas as pd
from joblib import load

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# %% setup
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

K = 5  # GMM components
EPOCHS = 200  # training
EARLY_PATIENCE = 20
EARLY_MIN_DELTA = 1e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW = 40
ADD_GMM_PROBS = True  # append GMM posteriors
ADD_VELOCITY = True  # append dX (first differences)
PER_SHOT_NORM = True  # per-shot zscore after global scaler

# Model config
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 3
DROPOUT = 0.10
MAX_LEN = 400  # positional encoding max length

SAVE_FINAL = "final_shot_transformer.pt"


# %% Training data setup
df = pd.read_csv("train.csv")
target_cols = ["angle", "depth", "left_right"]

cols_to_remove = ["id", "shot_id"] + target_cols
data = df.drop(columns=cols_to_remove, errors="ignore")

# drop participant_id from signals
signal_cols = [c for c in data.columns if c != "participant_id"]


def parse_row_to_shot(row, signal_cols_, keep_hands=True):
    """row: pd.Series; returns (T, n_signals) float array or None."""
    row_points = []
    hand_points = [
        col
        for col in signal_cols_
        if "right" in col
        and any(
            kw in col
            for kw in [
                "first_finger",
                "second_finger",
                "third_finger",
                "fourth_finger",
                "fifth_finger",
                "thumb",
                "pinky",
            ]
        )
    ]
    if not keep_hands:
        hand_points = signal_cols_

    for col in hand_points:
        s = str(row[col]).replace("nan", "None")
        try:
            arr = np.array(ast.literal_eval(s), dtype=float)
        except Exception:
            return None

        if np.isnan(arr).any():
            if np.isnan(arr).all():
                return None
            arr = (
                pd.Series(arr)
                .interpolate(method="linear", limit_direction="both")
                .values
            )

        row_points.append(arr)

    dp = np.array(row_points, dtype=float)  # (n_signals, T)
    if dp.ndim != 2:
        return None
    return dp.T  # (T, n_signals)


all_shots = []
kept_indices = []

for i in range(len(df)):
    shot = parse_row_to_shot(data.iloc[i], signal_cols, keep_hands=True)
    if shot is None:
        continue
    all_shots.append(shot)
    kept_indices.append(i)

if len(all_shots) == 0:
    raise RuntimeError("No valid train shots parsed.")

n_signals = all_shots[0].shape[1]
print(
    f"Kept train shots: {len(all_shots)} | n_signals={n_signals} | example T={all_shots[0].shape[0]}"
)

y_raw = df.loc[kept_indices, target_cols].reset_index(drop=True)

# scaling the output
sc_angle = load("scaler_angle.pkl")
sc_depth = load("scaler_depth.pkl")
sc_lr = load("scaler_left_right.pkl")


def scale_targets(y_df: pd.DataFrame) -> pd.DataFrame:
    y_angle = sc_angle.transform(y_df[["angle"]].values).ravel()
    y_depth = sc_depth.transform(y_df[["depth"]].values).ravel()
    y_lr = sc_lr.transform(y_df[["left_right"]].values).ravel()
    return pd.DataFrame({"angle": y_angle, "depth": y_depth, "left_right": y_lr})


y = scale_targets(y_raw)
train_mean_scaled = y.mean().to_dict()


# %% Model pieces
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B,T,d_model)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class ShotTransformer(nn.Module):
    def __init__(
        self, d_in, d_model=128, nhead=8, num_layers=3, dropout=0.1, max_len=400
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=d_in, out_channels=d_model, kernel_size=9, padding=4),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # attention pooling
        self.attn = nn.Linear(d_model, 1)

        self.angle_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.depth_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.lr_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x, mask):
        # x: (B,T,d_in), mask: (B,T) True valid
        x_conv = x.permute(0, 2, 1)  # (B,d_in,T)
        h = self.conv(x_conv).permute(0, 2, 1)  # (B,T,d_model)
        h = self.pos(h)

        pad_mask = ~mask
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        # attention pooling
        a = self.attn(h).squeeze(-1)  # (B,T)
        a = a.masked_fill(pad_mask, -1e9)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        pooled = (h * w).sum(dim=1)  # (B,d_model)

        angle = self.angle_head(pooled).squeeze(-1)
        depth = self.depth_head(pooled).squeeze(-1)
        lr = self.lr_head(pooled).squeeze(-1)
        return angle, depth, lr


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="min", save_path="best.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path
        self.best = None
        self.bad_epochs = 0
        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")

    def _is_improvement(self, current, best):
        if self.mode == "min":
            return current < (best - self.min_delta)
        return current > (best + self.min_delta)

    def step(self, current, model):
        if self.best is None:
            self.best = current
            torch.save(model.state_dict(), self.save_path)
            self.bad_epochs = 0
            return False

        if self._is_improvement(current, self.best):
            self.best = current
            torch.save(model.state_dict(), self.save_path)
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class ShotDataset(Dataset):
    def __init__(self, X_seq_list, y_df):
        self.X = X_seq_list
        self.y_angle = y_df["angle"].values.astype(np.float32)
        self.y_depth = y_df["depth"].values.astype(np.float32)
        self.y_lr = y_df["left_right"].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_angle[idx], self.y_depth[idx], self.y_lr[idx]


def collate_pad(batch):
    xs, angles, depths, lrs = zip(*batch)
    lengths = [x.shape[0] for x in xs]
    maxT = max(lengths)

    B = len(xs)
    d = xs[0].shape[1]

    x_pad = np.zeros((B, maxT, d), dtype=np.float32)
    mask = np.zeros((B, maxT), dtype=np.bool_)  # True valid

    for i, x in enumerate(xs):
        T = x.shape[0]
        x_pad[i, :T, :] = x
        mask[i, :T] = True

    return (
        torch.from_numpy(x_pad),
        torch.from_numpy(mask),
        torch.tensor(angles, dtype=torch.float32),
        torch.tensor(depths, dtype=torch.float32),
        torch.tensor(lrs, dtype=torch.float32),
    )


def collate_pad_infer(batch):
    xs = [b[0] for b in batch]
    lengths = [x.shape[0] for x in xs]
    maxT = max(lengths)

    B = len(xs)
    d = xs[0].shape[1]

    x_pad = np.zeros((B, maxT, d), dtype=np.float32)
    mask = np.zeros((B, maxT), dtype=np.bool_)

    for i, x in enumerate(xs):
        T = x.shape[0]
        x_pad[i, :T, :] = x
        mask[i, :T] = True

    return torch.from_numpy(x_pad), torch.from_numpy(mask)


# %% feature building
def build_features(
    shots_list,
    signal_cols_,
    scaler_,
    gmm_,
    window=40,
    add_probs=True,
    add_vel=True,
    per_shot_norm=True,
):
    """
    shots_list: list of RAW shots, each (T, n_signals)
    returns: list of feature arrays, each (T_win, d_in)
    """
    out = []

    for shot in shots_list:
        # 1) global scaling (trained on train timepoints)
        X = scaler_.transform(shot).astype(np.float32)  # (T, n_signals)

        # 2) optional per-shot zscore (helps cross-subject)
        if per_shot_norm:
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, keepdims=True) + 1e-6
            X = (X - mu) / sd

        feats = [X]

        # 3) optional velocity features (first differences)
        if add_vel:
            dX = np.diff(X, axis=0, prepend=X[:1])
            feats.append(dX.astype(np.float32))

        # 4) optional GMM posteriors for every frame
        if add_probs:
            R = gmm_.predict_proba(X).astype(np.float32)  # (T, K)
            feats.append(R)

        feat = np.concatenate(feats, axis=1).astype(np.float32)  # (T, d_in)

        out.append(feat)

    return out


# %% Loss + evaluation
huber = nn.SmoothL1Loss(beta=0.1)
mse = nn.MSELoss()


def multitask_loss(pa, pdd, plr, a, d, lr):
    return mse(pa, a) + mse(pdd, d) + mse(plr, lr)


def evaluate_mae(model, loader):
    model.eval()
    a_list, d_list, lr_list = [], [], []
    with torch.no_grad():
        for x, mask, a, d, lr in loader:
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            a, d, lr = a.to(DEVICE), d.to(DEVICE), lr.to(DEVICE)
            pa, pdd, plr = model(x, mask)
            a_list.append(torch.mean(torch.abs(pa - a)).item())
            d_list.append(torch.mean(torch.abs(pdd - d)).item())
            lr_list.append(torch.mean(torch.abs(plr - lr)).item())
    return float(np.mean(a_list) + np.mean(d_list) + np.mean(lr_list)) / 3.0


# %% LOSO CV (participant_id ONLY for splitting)

if "participant_id" not in df.columns:
    raise RuntimeError(
        "participant_id column not found in train.csv (needed for LOSO CV split)."
    )

subject_ids = df.loc[kept_indices, "participant_id"].reset_index(drop=True).values
unique_subjects = np.unique(subject_ids)
print(f"Found {len(unique_subjects)} unique subjects for LOSO CV.")


def run_fold(fold_id, train_idx, val_idx):
    torch.manual_seed(SEED + fold_id)
    np.random.seed(SEED + fold_id)

    shots_train = [all_shots[i] for i in train_idx]
    shots_val = [all_shots[i] for i in val_idx]
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)

    # Fit scaler + GMM on TRAIN timepoints
    X_train_all = np.vstack(shots_train)
    scaler_fold = StandardScaler()
    X_train_scaled = scaler_fold.fit_transform(X_train_all)

    gmm_fold = GaussianMixture(
        n_components=K,
        random_state=SEED + fold_id,
        covariance_type="full",
        reg_covar=1e-6,
    )
    gmm_fold.fit(X_train_scaled)

    # Build features
    X_train_feats = build_features(
        shots_train,
        signal_cols,
        scaler_fold,
        gmm_fold,
        window=WINDOW,
        add_probs=ADD_GMM_PROBS,
        add_vel=ADD_VELOCITY,
        per_shot_norm=PER_SHOT_NORM,
    )
    X_val_feats = build_features(
        shots_val,
        signal_cols,
        scaler_fold,
        gmm_fold,
        window=WINDOW,
        add_probs=ADD_GMM_PROBS,
        add_vel=ADD_VELOCITY,
        per_shot_norm=PER_SHOT_NORM,
    )

    d_in_fold = X_train_feats[0].shape[1]

    train_loader = DataLoader(
        ShotDataset(X_train_feats, y_train),
        batch_size=64,
        shuffle=True,
        collate_fn=collate_pad,
    )
    val_loader = DataLoader(
        ShotDataset(X_val_feats, y_val),
        batch_size=64,
        shuffle=False,
        collate_fn=collate_pad,
    )

    model = ShotTransformer(
        d_in=d_in_fold,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    early = EarlyStopping(
        patience=EARLY_PATIENCE,
        min_delta=EARLY_MIN_DELTA,
        mode="min",
        save_path=f"best_model_fold{fold_id}.pt",
    )

    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, mask, a, d, lr in train_loader:
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            a, d, lr = a.to(DEVICE), d.to(DEVICE), lr.to(DEVICE)

            pa, pdd, plr = model(x, mask)
            loss = multitask_loss(pa, pdd, plr, a, d, lr)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sched.step()

        val_mae = evaluate_mae(model, val_loader)

        if early.step(val_mae, model):
            break

    model.load_state_dict(
        torch.load(f"best_model_fold{fold_id}.pt", map_location=DEVICE)
    )
    return float(early.best)


fold_scores = []
for fold_id, holdout_subj in enumerate(unique_subjects):
    val_idx = np.where(subject_ids == holdout_subj)[0]
    train_idx = np.where(subject_ids != holdout_subj)[0]
    if len(val_idx) < 1 or len(train_idx) < 2:
        print(
            f"Skipping subject {holdout_subj} (train={len(train_idx)}, val={len(val_idx)})."
        )
        continue
    score = run_fold(fold_id, train_idx, val_idx)
    fold_scores.append(score)
    print(
        f"[Fold {fold_id+1:02d}/{len(unique_subjects)}] Holdout subj={holdout_subj} | best val MAE={score:.6f}"
    )

fold_scores = np.array(fold_scores, dtype=float)
print("\n===== LOSO CV Results =====")
print(f"Folds run: {len(fold_scores)} / {len(unique_subjects)}")
print(f"Mean best val MAE: {fold_scores.mean():.6f}")
print(
    f"Std  best val MAE: {fold_scores.std(ddof=1) if len(fold_scores)>1 else 0.0:.6f}"
)
print(f"Min/Max best val MAE: {fold_scores.min():.6f} / {fold_scores.max():.6f}")

# %% TRAIN ONCE MORE ON ALL DATA
rng = np.random.RandomState(SEED)
idx_all = np.arange(len(all_shots))
rng.shuffle(idx_all)

val_frac = 0.10
n_val = max(1, int(len(idx_all) * val_frac))
val_idx = idx_all[:n_val]
train_idx = idx_all[n_val:]

shots_train = [all_shots[i] for i in train_idx]
shots_val = [all_shots[i] for i in val_idx]
y_train = y.iloc[train_idx].reset_index(drop=True)
y_val = y.iloc[val_idx].reset_index(drop=True)

X_train_all = np.vstack(shots_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_all)

gmm = GaussianMixture(
    n_components=K,
    random_state=SEED,
    covariance_type="full",
    reg_covar=1e-6,
)
gmm.fit(X_train_scaled)

X_train_feats = build_features(
    shots_train,
    signal_cols,
    scaler,
    gmm,
    window=WINDOW,
    add_probs=ADD_GMM_PROBS,
    add_vel=ADD_VELOCITY,
    per_shot_norm=PER_SHOT_NORM,
)
X_val_feats = build_features(
    shots_val,
    signal_cols,
    scaler,
    gmm,
    window=WINDOW,
    add_probs=ADD_GMM_PROBS,
    add_vel=ADD_VELOCITY,
    per_shot_norm=PER_SHOT_NORM,
)

d_in = X_train_feats[0].shape[1]
print(f"\nFinal training d_in = {d_in}")

train_loader = DataLoader(
    ShotDataset(X_train_feats, y_train),
    batch_size=32,
    shuffle=True,
    collate_fn=collate_pad,
)
val_loader = DataLoader(
    ShotDataset(X_val_feats, y_val),
    batch_size=64,
    shuffle=False,
    collate_fn=collate_pad,
)

model = ShotTransformer(
    d_in=d_in,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    max_len=MAX_LEN,
).to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

early = EarlyStopping(
    patience=EARLY_PATIENCE,
    min_delta=EARLY_MIN_DELTA,
    mode="min",
    save_path=SAVE_FINAL,
)

for ep in range(1, EPOCHS + 1):
    model.train()
    for x, mask, a, d, lr in train_loader:
        x, mask = x.to(DEVICE), mask.to(DEVICE)
        a, d, lr = a.to(DEVICE), d.to(DEVICE), lr.to(DEVICE)

        pa, pdd, plr = model(x, mask)
        loss = multitask_loss(pa, pdd, plr, a, d, lr)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    sched.step()

    train_mae = evaluate_mae(model, train_loader)
    val_mae = evaluate_mae(model, val_loader)
    print(f"Final Epoch {ep:03d} | train MAE={train_mae:.6f} | val MAE={val_mae:.6f}")

    if early.step(val_mae, model):
        print(f"Early stopping at epoch {ep} (best val MAE={early.best:.6f})")
        break

model.load_state_dict(torch.load(SAVE_FINAL, map_location=DEVICE))
print(f"Loaded best final model from {SAVE_FINAL} (best val MAE={early.best:.6f})")

# %% TEST INFERENCE
test_df = pd.read_csv("test.csv")
test_data = test_df.drop(columns=["id", "shot_id"], errors="ignore")
signal_cols_test = [c for c in test_data.columns if c != "participant_id"]

test_shots = []
good_rows = []
bad_rows = []

for i in range(len(test_df)):
    shot = parse_row_to_shot(test_data.iloc[i], signal_cols_test, keep_hands=True)
    if shot is None:
        bad_rows.append(i)
        continue
    test_shots.append(shot)
    good_rows.append(i)

print(f"Test rows: {len(test_df)} | good: {len(good_rows)} | bad: {len(bad_rows)}")

# Build features using FINAL scaler+gmm
test_features = build_features(
    test_shots,
    signal_cols_test,
    scaler,
    gmm,
    window=WINDOW,
    add_probs=ADD_GMM_PROBS,
    add_vel=ADD_VELOCITY,
    per_shot_norm=PER_SHOT_NORM,
)

test_loader = DataLoader(
    [(x,) for x in test_features],
    batch_size=64,
    shuffle=False,
    collate_fn=collate_pad_infer,
)

model.eval()
pred_angle, pred_depth, pred_lr = [], [], []
with torch.no_grad():
    for x, mask in test_loader:
        x, mask = x.to(DEVICE), mask.to(DEVICE)
        a, d, lr = model(x, mask)
        pred_angle.append(a.cpu().numpy())
        pred_depth.append(d.cpu().numpy())
        pred_lr.append(lr.cpu().numpy())

pred_angle = np.concatenate(pred_angle)
pred_depth = np.concatenate(pred_depth)
pred_lr = np.concatenate(pred_lr)

# Put predictions back into full-length arrays (fill bad rows with train mean)
scaled_angle_full = np.full(len(test_df), train_mean_scaled["angle"], dtype=float)
scaled_depth_full = np.full(len(test_df), train_mean_scaled["depth"], dtype=float)
scaled_lr_full = np.full(len(test_df), train_mean_scaled["left_right"], dtype=float)

scaled_angle_full[good_rows] = pred_angle
scaled_depth_full[good_rows] = pred_depth
scaled_lr_full[good_rows] = pred_lr

# Clip to [0,1]
scaled_angle_full = np.clip(scaled_angle_full, 0.0, 1.0)
scaled_depth_full = np.clip(scaled_depth_full, 0.0, 1.0)
scaled_lr_full = np.clip(scaled_lr_full, 0.0, 1.0)

submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "scaled_angle": scaled_angle_full,
        "scaled_depth": scaled_depth_full,
        "scaled_left_right": scaled_lr_full,
    }
)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")

print(
    "Submission stds:",
    submission["scaled_angle"].std(),
    submission["scaled_depth"].std(),
    submission["scaled_left_right"].std(),
)

# %%
