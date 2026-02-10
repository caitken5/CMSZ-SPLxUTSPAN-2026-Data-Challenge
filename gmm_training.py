#%% GMM set up
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import ast
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def _parse_list_cell(cell) -> np.ndarray:
    """
    Convert a CSV cell into a float numpy array.
    """
    s = str(cell).replace("nan", "None")
    pts = ast.literal_eval(s)  # list (may include None)
    arr = np.array(pts, dtype=float)  # None -> np.nan
    return arr


def _interp_nans_1d(arr: np.ndarray) -> Optional[np.ndarray]:
    """
    Linearly interpolate NaNs in a 1D array.
    """
    if not np.isnan(arr).any():
        return arr

    nans = np.isnan(arr)
    not_nans = ~nans
    if not_nans.sum() == 0:
        return None  # all NaN

    arr = arr.copy()
    arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), arr[not_nans])
    return arr


def load_shots_from_csv(
    csv_path: str | Path,
    cols_to_remove: List[str],
    id_col: str = "participant_id",
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load shots from CSV.

    Returns
    -------
    shots : list of arrays
        Each element has shape (n_signals, T)
    valid_row_idx : list of int
        Row indices in the original dataframe corresponding to kept shots
    signal_cols : list of str
        Column names used as signals
    """
    df = pd.read_csv(csv_path)

    # Drop non-signal columns (targets, ids, etc.)
    data = df.drop(columns=[c for c in cols_to_remove if c in df.columns], errors="ignore")

    # Signal columns are everything except participant_id (or whatever id_col is)
    signal_cols = [c for c in data.columns if c != id_col]

    shots: List[np.ndarray] = []
    valid_row_idx: List[int] = []

    for i in range(len(data)):
        row = data.iloc[i]

        row_signals: List[np.ndarray] = []
        bad_row = False
        T_ref: Optional[int] = None

        for col in signal_cols:
            arr = _parse_list_cell(row[col])
            arr = _interp_nans_1d(arr)
            if arr is None:
                bad_row = True
                break

            row_signals.append(arr)

        if bad_row:
            continue

        dp = np.stack(row_signals, axis=0).astype(float)  # (n_signals, T)
        shots.append(dp)
        valid_row_idx.append(i)

    print(f"Kept {len(shots)} / {len(data)} shots after parsing + NaN handling.")
    print(f"n_signals = {len(signal_cols)}")
    if len(shots) > 0:
        print(f"Example shot shape = {shots[0].shape}  (n_signals, T)")

    return shots, valid_row_idx, signal_cols


def stack_timepoints(shots: List[np.ndarray]) -> np.ndarray:
    """
    Convert list of shots (n_signals, T) into a 2D matrix X (sum_T, n_signals)
    by stacking timepoints across shots. (Prep for GMM training)
    """
    X = np.vstack([dp.T for dp in shots])
    return X


def fit_scaler_gmm(X: np.ndarray, n_components: int, seed: int) -> Tuple[StandardScaler, GaussianMixture]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_components, random_state=seed)
    gmm.fit(X_scaled)
    return scaler, gmm


def export_per_shot_outputs(
    shots: List[np.ndarray],
    scaler: StandardScaler,
    gmm: GaussianMixture,
) -> None:
    """
    For each shot, exports:
      - labels: shape (T,)
      - proba : shape (T, K)
    where K = n_components
    """
    for idx, dp in enumerate(shots):
        X_shot = dp.T  # (T, n_signals)
        Xs = scaler.transform(X_shot)

        labels.append(gmm.predict(Xs))  # (T,)
        probs.append(gmm.predict_proba(Xs))  # (T, K)


    return labels, probs


#%% Use the above functions on train data
csv_path = "train.csv"
cols_to_remove = ["id", "shot_id", "angle", "depth", "left_right"]
shots, valid_row_idx, signal_cols = load_shots_from_csv(csv_path, cols_to_remove)
X = stack_timepoints(shots)
scaler, gmm = fit_scaler_gmm(X, n_components=5, seed=42)
export_per_shot_outputs(shots, scaler, gmm)


