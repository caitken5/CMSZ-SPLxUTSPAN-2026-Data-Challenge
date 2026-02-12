# %% Libraries
import math
import ast
import numpy as np
import pandas as pd
from joblib import load
from typing import List, Tuple

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# %% setup
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

K = 5  # GMM components
EPOCHS = 200
EARLY_PATIENCE = 20
EARLY_MIN_DELTA = 1e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shapelet config
N_SHAPELETS_PER_PHASE = 5  # Extract 5 prototypes per phase
SHAPELET_LENGTHS = [10, 20, 30]  # Multiple temporal scales
N_PHASES = 4

# Model config
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 3
DROPOUT = 0.10
MAX_LEN = 400

SAVE_FINAL = "final_shapelet_transformer.pt"


# %% Training data setup
df = pd.read_csv("train.csv")
target_cols = ["angle", "depth", "left_right"]

cols_to_remove = ["id", "shot_id"] + target_cols
data = df.drop(columns=cols_to_remove, errors="ignore")

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

    dp = np.array(row_points, dtype=float)
    if dp.ndim != 2:
        return None
    return dp.T


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
print(f"Kept train shots: {len(all_shots)} | n_signals={n_signals}")

y_raw = df.loc[kept_indices, target_cols].reset_index(drop=True)

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


# %% Shapelet Extraction Functions
def segment_shot_by_gmm(shot: np.ndarray, gmm, scaler, n_phases: int = 4):
    """Segment shot into phases using GMM labels"""
    X_shot = scaler.transform(shot)
    labels = gmm.predict(X_shot)
    
    # Find phase transitions
    phase_starts = [0]
    current_label = labels[0]
    
    for t in range(1, len(labels)):
        if labels[t] != current_label:
            phase_starts.append(t)
            current_label = labels[t]
    
    # Map to n_phases
    T = len(labels)
    if len(phase_starts) < n_phases:
        # Split longest phase
        while len(phase_starts) < n_phases:
            phase_lengths = []
            for i in range(len(phase_starts)):
                end = phase_starts[i+1] if i+1 < len(phase_starts) else T
                phase_lengths.append((end - phase_starts[i], i))
            phase_lengths.sort(reverse=True)
            longest_len, longest_idx = phase_lengths[0]
            split_point = phase_starts[longest_idx] + longest_len // 2
            phase_starts.append(split_point)
            phase_starts.sort()
    
    # Build boundaries
    boundaries = []
    for i in range(n_phases):
        start = phase_starts[i] if i < len(phase_starts) else T-1
        end = phase_starts[i+1] if i+1 < len(phase_starts) else T
        boundaries.append((start, min(end, T)))
    
    # Extract phase data
    phases = []
    for start, end in boundaries:
        if end > start:
            phases.append(shot[start:end, :])
        else:
            phases.append(shot[start:start+1, :])
    
    return phases, boundaries


def extract_shapelets_from_phase(
    phase_shots: List[np.ndarray],
    length: int,
    n_shapelets: int
) -> List[np.ndarray]:
    """
    Extract representative shapelets from a collection of phase sequences.
    Uses k-means like approach with DTW.
    """
    if len(phase_shots) == 0:
        return []
    
    # Collect all valid candidates
    candidates = []
    for shot in phase_shots:
        T = shot.shape[0]
        if T < length:
            continue
        # Extract sliding windows
        for start in range(0, T - length + 1, max(1, length // 4)):
            candidates.append(shot[start:start+length, :])
    
    if len(candidates) == 0:
        return []
    
    if len(candidates) <= n_shapelets:
        return candidates
    
    # Simple k-means with DTW (simplified for speed)
    # Randomly sample n_shapelets as initial prototypes
    np.random.seed(SEED)
    indices = np.random.choice(len(candidates), n_shapelets, replace=False)
    prototypes = [candidates[i] for i in indices]
    
    # One iteration of refinement
    clusters = [[] for _ in range(n_shapelets)]
    for cand in candidates:
        # Find nearest prototype
        min_dist = float('inf')
        min_idx = 0
        for i, proto in enumerate(prototypes):
            dist, _ = fastdtw(cand, proto, dist=euclidean)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        clusters[min_idx].append(cand)
    
    # Update prototypes (use medoid)
    for i in range(n_shapelets):
        if len(clusters[i]) == 0:
            continue
        cluster = clusters[i]
        # Find medoid (element with min sum distance to others)
        min_sum_dist = float('inf')
        medoid = cluster[0]
        for cand in cluster[:min(20, len(cluster))]:  # Limit for speed
            sum_dist = sum([fastdtw(cand, other, dist=euclidean)[0] 
                           for other in cluster[:min(20, len(cluster))]])
            if sum_dist < min_sum_dist:
                min_sum_dist = sum_dist
                medoid = cand
        prototypes[i] = medoid
    
    return prototypes


def extract_all_shapelets(
    shots: List[np.ndarray],
    gmm,
    scaler,
    n_phases: int = 4,
    n_shapelets_per_phase: int = 5,
    lengths: List[int] = [10, 20, 30]
):
    """
    Extract shapelets for each phase at multiple temporal scales.
    
    Returns: dict mapping (phase_id, length) -> list of shapelet arrays
    """
    # Segment all shots
    all_phases = [[] for _ in range(n_phases)]
    for shot in shots:
        phases, _ = segment_shot_by_gmm(shot, gmm, scaler, n_phases)
        for p, phase_data in enumerate(phases):
            all_phases[p].append(phase_data)
    
    # Extract shapelets for each phase and length
    shapelets = {}
    for p in range(n_phases):
        for length in lengths:
            key = (p, length)
            shapelets[key] = extract_shapelets_from_phase(
                all_phases[p], length, n_shapelets_per_phase
            )
            print(f"Phase {p}, length {length}: extracted {len(shapelets[key])} shapelets")
    
    return shapelets


def compute_shapelet_features(
    shot: np.ndarray,
    shapelets: dict,
    gmm,
    scaler,
    n_phases: int = 4
) -> np.ndarray:
    """
    Compute DTW distances from shot to all shapelets.
    Returns vector of min DTW distances.
    """
    phases, _ = segment_shot_by_gmm(shot, gmm, scaler, n_phases)
    
    features = []
    for p in range(n_phases):
        phase_data = phases[p]
        for length in SHAPELET_LENGTHS:
            key = (p, length)
            if key not in shapelets or len(shapelets[key]) == 0:
                # No shapelets for this phase/length
                features.extend([0.0] * N_SHAPELETS_PER_PHASE)
                continue
            
            # Compute min DTW to each shapelet
            for shapelet in shapelets[key]:
                dist, _ = fastdtw(phase_data, shapelet, dist=euclidean)
                features.append(dist)
    
    return np.array(features, dtype=np.float32)


# %% Model
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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]


class ShapeletTransformer(nn.Module):
    """
    Transformer that combines:
    1. Raw temporal signals (processed by transformer)
    2. Shapelet DTW distances (global features)
    """
    def __init__(
        self, d_temporal, d_shapelet, d_model=128, nhead=8, 
        num_layers=3, dropout=0.1, max_len=400
    ):
        super().__init__()
        
        # Temporal processing
        self.conv = nn.Sequential(
            nn.Conv1d(d_temporal, d_model, kernel_size=9, padding=4),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        # Attention pooling
        self.attn = nn.Linear(d_model, 1)
        
        # Shapelet feature processing
        self.shapelet_proj = nn.Sequential(
            nn.Linear(d_shapelet, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task heads
        self.angle_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.depth_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.lr_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x_temporal, mask, x_shapelet):
        # Process temporal data
        x_conv = x_temporal.permute(0, 2, 1)
        h = self.conv(x_conv).permute(0, 2, 1)
        h = self.pos(h)
        
        pad_mask = ~mask
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        
        # Attention pooling
        a = self.attn(h).squeeze(-1)
        a = a.masked_fill(pad_mask, -1e9)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        temporal_pooled = (h * w).sum(dim=1)
        
        # Process shapelet features
        shapelet_embed = self.shapelet_proj(x_shapelet)
        
        # Fuse
        combined = torch.cat([temporal_pooled, shapelet_embed], dim=1)
        fused = self.fusion(combined)
        
        # Predictions
        angle = self.angle_head(fused).squeeze(-1)
        depth = self.depth_head(fused).squeeze(-1)
        lr = self.lr_head(fused).squeeze(-1)
        
        return angle, depth, lr


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="min", save_path="best.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path
        self.best = None
        self.bad_epochs = 0

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


class ShapeletDataset(Dataset):
    def __init__(self, temporal_list, shapelet_list, y_df):
        self.temporal = temporal_list
        self.shapelet = shapelet_list
        self.y_angle = y_df["angle"].values.astype(np.float32)
        self.y_depth = y_df["depth"].values.astype(np.float32)
        self.y_lr = y_df["left_right"].values.astype(np.float32)

    def __len__(self):
        return len(self.temporal)

    def __getitem__(self, idx):
        return (
            self.temporal[idx],
            self.shapelet[idx],
            self.y_angle[idx],
            self.y_depth[idx],
            self.y_lr[idx]
        )


def collate_shapelet_batch(batch):
    temporal_batch, shapelet_batch, angles, depths, lrs = zip(*batch)
    
    B = len(temporal_batch)
    T_max = max(x.shape[0] for x in temporal_batch)
    d_temporal = temporal_batch[0].shape[1]
    
    # Pad temporal
    x_pad = torch.zeros(B, T_max, d_temporal, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.bool)
    
    for i, x in enumerate(temporal_batch):
        T = x.shape[0]
        x_pad[i, :T, :] = torch.tensor(x, dtype=torch.float32)
        mask[i, :T] = True
    
    # Stack shapelet features
    x_shapelet = torch.tensor(np.stack(shapelet_batch), dtype=torch.float32)
    
    return (
        x_pad,
        mask,
        x_shapelet,
        torch.tensor(angles, dtype=torch.float32),
        torch.tensor(depths, dtype=torch.float32),
        torch.tensor(lrs, dtype=torch.float32),
    )


def collate_shapelet_batch_infer(batch):
    temporal_batch, shapelet_batch = zip(*[b[:2] for b in batch])
    
    B = len(temporal_batch)
    T_max = max(x.shape[0] for x in temporal_batch)
    d_temporal = temporal_batch[0].shape[1]
    
    x_pad = torch.zeros(B, T_max, d_temporal, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.bool)
    
    for i, x in enumerate(temporal_batch):
        T = x.shape[0]
        x_pad[i, :T, :] = torch.tensor(x, dtype=torch.float32)
        mask[i, :T] = True
    
    x_shapelet = torch.tensor(np.stack(shapelet_batch), dtype=torch.float32)
    
    return x_pad, mask, x_shapelet


# %% Feature building
def build_features(shots_list, scaler, per_shot_norm=True):
    """Basic feature building with optional per-shot normalization"""
    out = []
    for shot in shots_list:
        X = scaler.transform(shot).astype(np.float32)
        
        if per_shot_norm:
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, keepdims=True) + 1e-6
            X = (X - mu) / sd
        
        # Add velocity
        dX = np.diff(X, axis=0, prepend=X[:1])
        feat = np.concatenate([X, dX], axis=1).astype(np.float32)
        out.append(feat)
    
    return out


# %% Loss
mse = nn.MSELoss()


def multitask_loss(pa, pdd, plr, a, d, lr):
    return mse(pa, a) + mse(pdd, d) + mse(plr, lr)


def evaluate_mae(model, loader):
    model.eval()
    a_list, d_list, lr_list = [], [], []
    with torch.no_grad():
        for x_temp, mask, x_shape, a, d, lr in loader:
            x_temp = x_temp.to(DEVICE)
            mask = mask.to(DEVICE)
            x_shape = x_shape.to(DEVICE)
            a, d, lr = a.to(DEVICE), d.to(DEVICE), lr.to(DEVICE)
            
            pa, pdd, plr = model(x_temp, mask, x_shape)
            a_list.append(torch.mean(torch.abs(pa - a)).item())
            d_list.append(torch.mean(torch.abs(pdd - d)).item())
            lr_list.append(torch.mean(torch.abs(plr - lr)).item())
    return float(np.mean(a_list) + np.mean(d_list) + np.mean(lr_list)) / 3.0


# %% LOSO CV
if "participant_id" not in df.columns:
    raise RuntimeError("participant_id column not found")

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

    # Fit scaler + GMM
    X_train_all = np.vstack(shots_train)
    scaler_fold = StandardScaler()
    X_train_scaled = scaler_fold.fit_transform(X_train_all)

    gmm_fold = GaussianMixture(
        n_components=K, random_state=SEED + fold_id,
        covariance_type="full", reg_covar=1e-6
    )
    gmm_fold.fit(X_train_scaled)

    # Extract shapelets from TRAIN
    print(f"Fold {fold_id}: Extracting shapelets...")
    shapelets = extract_all_shapelets(
        shots_train, gmm_fold, scaler_fold,
        n_phases=N_PHASES,
        n_shapelets_per_phase=N_SHAPELETS_PER_PHASE,
        lengths=SHAPELET_LENGTHS
    )

    # Build features
    X_train_temp = build_features(shots_train, scaler_fold, per_shot_norm=True)
    X_val_temp = build_features(shots_val, scaler_fold, per_shot_norm=True)

    # Compute shapelet features
    print(f"Fold {fold_id}: Computing shapelet features...")
    X_train_shape = [
        compute_shapelet_features(shot, shapelets, gmm_fold, scaler_fold, N_PHASES)
        for shot in shots_train
    ]
    X_val_shape = [
        compute_shapelet_features(shot, shapelets, gmm_fold, scaler_fold, N_PHASES)
        for shot in shots_val
    ]

    d_temporal = X_train_temp[0].shape[1]
    d_shapelet = X_train_shape[0].shape[0]

    train_loader = DataLoader(
        ShapeletDataset(X_train_temp, X_train_shape, y_train),
        batch_size=64, shuffle=True, collate_fn=collate_shapelet_batch
    )
    val_loader = DataLoader(
        ShapeletDataset(X_val_temp, X_val_shape, y_val),
        batch_size=64, shuffle=False, collate_fn=collate_shapelet_batch
    )

    model = ShapeletTransformer(
        d_temporal=d_temporal,
        d_shapelet=d_shapelet,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        max_len=MAX_LEN
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    early = EarlyStopping(
        patience=EARLY_PATIENCE, min_delta=EARLY_MIN_DELTA,
        mode="min", save_path=f"best_shapelet_fold{fold_id}.pt"
    )

    for ep in range(1, EPOCHS + 1):
        model.train()
        for x_temp, mask, x_shape, a, d, lr in train_loader:
            x_temp = x_temp.to(DEVICE)
            mask = mask.to(DEVICE)
            x_shape = x_shape.to(DEVICE)
            a, d, lr = a.to(DEVICE), d.to(DEVICE), lr.to(DEVICE)

            pa, pdd, plr = model(x_temp, mask, x_shape)
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
        torch.load(f"best_shapelet_fold{fold_id}.pt", map_location=DEVICE)
    )
    return float(early.best), shapelets, scaler_fold, gmm_fold


fold_scores = []
for fold_id, holdout_subj in enumerate(unique_subjects):
    val_idx = np.where(subject_ids == holdout_subj)[0]
    train_idx = np.where(subject_ids != holdout_subj)[0]
    if len(val_idx) < 1 or len(train_idx) < 2:
        print(f"Skipping subject {holdout_subj}")
        continue
    score, _, _, _ = run_fold(fold_id, train_idx, val_idx)
    fold_scores.append(score)
    print(f"[Fold {fold_id+1:02d}] Holdout={holdout_subj} | MAE={score:.6f}")

fold_scores = np.array(fold_scores, dtype=float)
print("\n===== SHAPELET LOSO CV Results =====")
print(f"Mean MAE: {fold_scores.mean():.6f}")
print(f"Std MAE: {fold_scores.std(ddof=1) if len(fold_scores)>1 else 0.0:.6f}")

# %% TRAIN FINAL MODEL
print("\n===== Training final model on all data =====")
rng = np.random.RandomState(SEED)
idx_all = np.arange(len(all_shots))
rng.shuffle(idx_all)

n_val = max(1, int(len(idx_all) * 0.10))
val_idx = idx_all[:n_val]
train_idx = idx_all[n_val:]

shots_train = [all_shots[i] for i in train_idx]
shots_val = [all_shots[i] for i in val_idx]
y_train = y.iloc[train_idx].reset_index(drop=True)
y_val = y.iloc[val_idx].reset_index(drop=True)

X_train_all = np.vstack(shots_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_all)

gmm = GaussianMixture(n_components=K, random_state=SEED, covariance_type="full", reg_covar=1e-6)
gmm.fit(X_train_scaled)

print("Extracting final shapelets...")
shapelets = extract_all_shapelets(
    shots_train, gmm, scaler, N_PHASES, N_SHAPELETS_PER_PHASE, SHAPELET_LENGTHS
)

X_train_temp = build_features(shots_train, scaler, per_shot_norm=True)
X_val_temp = build_features(shots_val, scaler, per_shot_norm=True)

print("Computing final shapelet features...")
X_train_shape = [
    compute_shapelet_features(shot, shapelets, gmm, scaler, N_PHASES)
    for shot in shots_train
]
X_val_shape = [
    compute_shapelet_features(shot, shapelets, gmm, scaler, N_PHASES)
    for shot in shots_val
]

d_temporal = X_train_temp[0].shape[1]
d_shapelet = X_train_shape[0].shape[0]

train_loader = DataLoader(
    ShapeletDataset(X_train_temp, X_train_shape, y_train),
    batch_size=32, shuffle=True, collate_fn=collate_shapelet_batch
)
val_loader = DataLoader(
    ShapeletDataset(X_val_temp, X_val_shape, y_val),
    batch_size=64, shuffle=False, collate_fn=collate_shapelet_batch
)

model = ShapeletTransformer(
    d_temporal=d_temporal, d_shapelet=d_shapelet,
    d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
    dropout=DROPOUT, max_len=MAX_LEN
).to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

early = EarlyStopping(patience=EARLY_PATIENCE, min_delta=EARLY_MIN_DELTA, mode="min", save_path=SAVE_FINAL)

for ep in range(1, EPOCHS + 1):
    model.train()
    for x_temp, mask, x_shape, a, d, lr in train_loader:
        x_temp = x_temp.to(DEVICE)
        mask = mask.to(DEVICE)
        x_shape = x_shape.to(DEVICE)
        a, d, lr = a.to(DEVICE), d.to(DEVICE), lr.to(DEVICE)

        pa, pdd, plr = model(x_temp, mask, x_shape)
        loss = multitask_loss(pa, pdd, plr, a, d, lr)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    sched.step()
    
    train_mae = evaluate_mae(model, train_loader)
    val_mae = evaluate_mae(model, val_loader)
    print(f"Epoch {ep:03d} | train MAE={train_mae:.6f} | val MAE={val_mae:.6f}")

    if early.step(val_mae, model):
        print(f"Early stopping at epoch {ep}")
        break

model.load_state_dict(torch.load(SAVE_FINAL, map_location=DEVICE))
print(f"Loaded best model (val MAE={early.best:.6f})")

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

print(f"Test: {len(test_df)} total | {len(good_rows)} good | {len(bad_rows)} bad")

test_temp = build_features(test_shots, scaler, per_shot_norm=True)

print("Computing test shapelet features...")
test_shape = [
    compute_shapelet_features(shot, shapelets, gmm, scaler, N_PHASES)
    for shot in test_shots
]

test_loader = DataLoader(
    [(temp, shape) for temp, shape in zip(test_temp, test_shape)],
    batch_size=64, shuffle=False, collate_fn=collate_shapelet_batch_infer
)

model.eval()
pred_angle, pred_depth, pred_lr = [], [], []
with torch.no_grad():
    for x_temp, mask, x_shape in test_loader:
        x_temp = x_temp.to(DEVICE)
        mask = mask.to(DEVICE)
        x_shape = x_shape.to(DEVICE)
        
        a, d, lr = model(x_temp, mask, x_shape)
        pred_angle.append(a.cpu().numpy())
        pred_depth.append(d.cpu().numpy())
        pred_lr.append(lr.cpu().numpy())

pred_angle = np.concatenate(pred_angle)
pred_depth = np.concatenate(pred_depth)
pred_lr = np.concatenate(pred_lr)

# Fill bad rows
scaled_angle_full = np.full(len(test_df), train_mean_scaled["angle"], dtype=float)
scaled_depth_full = np.full(len(test_df), train_mean_scaled["depth"], dtype=float)
scaled_lr_full = np.full(len(test_df), train_mean_scaled["left_right"], dtype=float)

scaled_angle_full[good_rows] = pred_angle
scaled_depth_full[good_rows] = pred_depth
scaled_lr_full[good_rows] = pred_lr

scaled_angle_full = np.clip(scaled_angle_full, 0.0, 1.0)
scaled_depth_full = np.clip(scaled_depth_full, 0.0, 1.0)
scaled_lr_full = np.clip(scaled_lr_full, 0.0, 1.0)

submission = pd.DataFrame({
    "id": test_df["id"].values,
    "scaled_angle": scaled_angle_full,
    "scaled_depth": scaled_depth_full,
    "scaled_left_right": scaled_lr_full,
})
submission.to_csv("submission_shapelet.csv", index=False)
print("Wrote submission_shapelet.csv")

# %%