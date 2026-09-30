"""Public Bonn data, analytical anchors and train/validation/test partitions."""
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.model_selection import GroupShuffleSplit, train_test_split

FEATURES = ["latitude", "longitude", "gw_distance_rounded_km"]
BASELINES = ["fspl", "ldpl_oulu", "okumura_hata"]
CLASSICAL_BASELINES = ["fspl", "ldpl_oulu", "okumura_hata", "winnerplus"]
FREQUENCY_MHZ = 868.0
SENSOR_HEIGHT_M = 2.0
EIRP_RX_GAIN_LOSS_DB = 15.0


def download(folder):
    folder.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/mclab-hbrs/lora-bonn/74d9e29a55eae6ba0909738e9f623f6e7219e902/survey"
    for name in ("samples.csv", "gateways.csv"):
        path = folder / name
        if not path.exists():
            print(f"Downloading {name}...", flush=True)
            with urlopen(f"{base}/{name}", timeout=120) as response:
                payload = response.read()
            temp = path.with_suffix(".tmp")
            temp.write_bytes(payload)
            temp.replace(path)


def load_measurements(samples_path: Path, gateways_path: Path) -> tuple[pd.DataFrame, dict]:
    samples = pd.read_csv(samples_path, dtype={"gw": str, "pkt_number": str})
    gateways = pd.read_csv(gateways_path, dtype={"id": str}).set_index("id")
    required_samples = {"latitude", "longitude", "transceived_at", "pkt_number", "rssi", "snr", "gw"}
    required_gateways = {"latitude", "longitude", "antenna_height"}
    if required_samples - set(samples) or required_gateways - set(gateways):
        raise ValueError("Input CSVs lack required Bonn survey columns")
    if gateways.index.has_duplicates:
        raise ValueError("Gateway IDs must be unique")

    raw_rows = len(samples)
    samples = samples.join(
        gateways.rename(columns={"latitude": "gw_latitude", "longitude": "gw_longitude"}),
        on="gw", validate="many_to_one"
    )
    numeric = ["latitude", "longitude", "rssi", "snr", "gw_latitude", "gw_longitude", "antenna_height"]
    samples[numeric] = samples[numeric].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(samples[numeric]).all(axis=1)
    valid &= samples.latitude.between(-90, 90) & samples.gw_latitude.between(-90, 90)
    valid &= samples.longitude.between(-180, 180) & samples.gw_longitude.between(-180, 180)
    valid &= samples.antenna_height.gt(0)
    valid &= samples[["transceived_at", "pkt_number", "gw"]].notna().all(axis=1)
    samples = samples.loc[valid].copy()

    # Round to the nearest metre before forming the ML distance feature, as in
    # the original notebook. Keep metres for the physical formulas.
    samples["gw_distance_m"] = [
        round(geodesic((lat, lon), (gw_lat, gw_lon)).meters)
        for lat, lon, gw_lat, gw_lon in zip(
            samples.latitude, samples.longitude, samples.gw_latitude, samples.gw_longitude
        )
    ]
    samples = samples.loc[samples.gw_distance_m.gt(0)].copy()
    samples["gw_distance_rounded_km"] = (samples.gw_distance_m / 1000).round(2)
    samples["log_distance_km"] = np.log10(samples.gw_distance_m / 1000)
    samples["rpp"] = samples.rssi + np.minimum(samples.snr, 0)
    samples["path_loss"] = EIRP_RX_GAIN_LOSS_DB - samples.rpp

    distance_m = samples.gw_distance_m.to_numpy(dtype=float)
    distance_km = distance_m / 1000
    height_m = samples.antenna_height.to_numpy(dtype=float)
    log_f = np.log10(FREQUENCY_MHZ)
    log_h = np.log10(height_m)
    samples["fspl"] = 20 * np.log10(distance_m) + 20 * log_f - 27.55
    samples["ldpl_oulu"] = 128.95 + 23.2 * np.log10(distance_km)
    receiver_correction = 0.8 + (1.1 * log_f - 0.7) * SENSOR_HEIGHT_M - 1.56 * log_f
    samples["okumura_hata"] = (
        69.55 + 26.16 * log_f - 13.82 * log_h - receiver_correction
        + (44.9 - 6.55 * log_h) * np.log10(distance_km)
    )
    samples["winnerplus"] = (
        (44.9 - 6.55 * log_h) * np.log10(distance_m)
        + 5.83 * log_h + 16.33 + 26.16 * np.log10(FREQUENCY_MHZ / 1000)
    )
    metadata = {
        "source_rows": raw_rows,
        "used_rows": len(samples),
        "excluded_rows": raw_rows - len(samples),
        "target": "15 - (rssi + min(snr, 0)) dB",
        "features": FEATURES,
        "physical_baselines": BASELINES,
        "additional_classical_baselines": [name for name in CLASSICAL_BASELINES if name not in BASELINES],
    }
    return samples, metadata

def fit_ldpl_bonn_train(df: pd.DataFrame, train_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit the Bonn LDPL on training-bin medians only and predict every row.

    This mirrors the source study's equal-bin curve fitting while preventing the
    validation and test targets from determining the local LDPL parameters.
    """
    train = df.iloc[train_idx]
    grouped = train.groupby("gw_distance_rounded_km", as_index=False).agg(
        path_loss=("path_loss", "median")
    )
    grouped = grouped.loc[grouped.gw_distance_rounded_km.gt(0)]
    design = np.column_stack([
        np.ones(len(grouped)), 10 * np.log10(grouped.gw_distance_rounded_km.to_numpy())
    ])
    pl_1km, exponent = np.linalg.lstsq(design, grouped.path_loss.to_numpy(), rcond=None)[0]
    prediction = pl_1km + 10 * exponent * np.log10(df.gw_distance_m.to_numpy() / 1000)
    return prediction, {"pl_at_1km_db": float(pl_1km), "path_loss_exponent": float(exponent),
                        "training_distance_bins": len(grouped)}

def packet_groups(df: pd.DataFrame) -> pd.Series:
    """Proxy transmission key; the public timestamp has date precision only."""
    return (df.transceived_at.astype(str) + "/" + df.pkt_number.astype(str)
            + "/" + df.latitude.astype(str) + "/" + df.longitude.astype(str))

def spatial_groups(df: pd.DataFrame) -> pd.Series:
    """Original 500 m grid, defined on the complete campaign.

    Compute once on the full frame and subset the labels afterwards: recomputing
    the reference latitude on a test subset moves the east/west tile boundaries.
    Tuple labels preserve the ordering used by the original GroupShuffleSplit.
    """
    latitude_m = df.latitude.to_numpy() * 111_320
    longitude_m = df.longitude.to_numpy() * 111_320 * np.cos(np.deg2rad(df.latitude.mean()))
    east = np.floor(longitude_m / 500).astype(int)
    north = np.floor(latitude_m / 500).astype(int)
    return pd.Series(list(zip(east, north)), index=df.index)

def split_rows(df: pd.DataFrame, method: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(df))
    if method == "random":
        train, held = train_test_split(indices, test_size=0.25, random_state=seed)
        validation, test = train_test_split(held, test_size=0.60, random_state=seed)
    elif method in {"packet", "spatial_500m"}:
        # A packet can be received by several gateways; keep those receptions
        # together. Packet IDs can recur within a day, so include GPS position.
        if method == "packet":
            groups = packet_groups(df)
        else:
            groups = spatial_groups(df)
        first = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_pos, held_pos = next(first.split(indices, groups=groups))
        train, held = indices[train_pos], indices[held_pos]
        second = GroupShuffleSplit(n_splits=1, test_size=0.60, random_state=seed)
        val_pos, test_pos = next(second.split(held, groups=groups.iloc[held_pos]))
        validation, test = held[val_pos], held[test_pos]
    else:
        raise ValueError(f"Unknown split method: {method}")
    return train, validation, test
