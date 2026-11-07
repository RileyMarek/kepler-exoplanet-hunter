"""
KTDG - Kepler DR24 TCE catalog cleaning and train/val/test split.

Usage:
    py prepare_data.py <path_to_dr24_csv> [--outdir out]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Label mapping. PC = planet candidate (positive).
# AFP = astrophysical false positive, NTP = non-transiting phenomenon (both negative).
# UNK = unvetted, dropped.
LABEL_MAP = {"PC": 1, "AFP": 0, "NTP": 0}

# Physical measurement columns we keep as raw features.
# Deliberately excludes every av_* column except the label, and all *_err columns.
CANDIDATE_FEATURES = [
    "tce_period",        # orbital period, days
    "tce_duration",      # transit duration, hours
    "tce_depth",         # transit depth, ppm
    "tce_model_snr",     # signal to noise
    "tce_prad",          # planet radius, Earth radii
    "tce_impact",        # impact parameter (grazing-ness)
    "tce_ror",           # planet/star radius ratio
    "tce_dor",           # semimajor axis / stellar radius
    "tce_bin_oedp_stat",  # odd/even depth difference statistic
    "tce_albedo_stat",   # secondary eclipse: albedo significance
    "tce_ptemp_stat",    # secondary eclipse: planet temperature significance
    "tce_max_mult_ev",   # multiple event statistic
    "tce_num_transits",  # how many transits were stacked
    "tce_robstat",       # robust detection statistic
    "tce_steff",         # stellar effective temperature
    "tce_slogg",         # stellar surface gravity
    "tce_sradius",       # stellar radius
    "tce_dicco_msky",    # centroid offset, difference image vs out-of-transit
    "tce_dikco_msky",    # centroid offset, difference image vs KIC position
]

# Columns needed downstream for phase folding, not used as model features.
PASSTHROUGH = ["kepid", "tce_plnt_num", "tce_period", "tce_time0bk", "tce_duration"]


def load(path):
    """Read the archive CSV, skipping its '#' comment header."""
    df = pd.read_csv(path, comment="#", low_memory=False)
    print(f"loaded {path}")
    print(f"  {len(df):,} rows, {len(df.columns)} columns, {df.kepid.nunique():,} stars")
    return df


def check_label(df):
    if "av_training_set" not in df.columns:
        sys.exit(
            "\nERROR: no 'av_training_set' column.\n"
            "You downloaded the archive's default column set, which has no labels.\n"
            "Re-download the full table:\n"
            "  https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
            "?query=select+*+from+q1_q17_dr24_tce&format=csv\n"
        )


def drop_leaky(df):
    """Remove the Robovetter's own output. av_training_set comes from the same
    vetting process, so these columns are effectively the answer key."""
    leaky = [c for c in df.columns
             if c.lower().startswith("av_") and c != "av_training_set"]
    if leaky:
        print(f"\ndropped {len(leaky)} leaky Robovetter columns:")
        print("  " + ", ".join(leaky))
        df = df.drop(columns=leaky)
    return df


def clean(df):
    """Drop rogue/unlabeled rows, map labels, select features."""
    n0 = len(df)

    # DR24 flags some TCEs as rogue (known pipeline artifacts). Drop them.
    if "tce_rogue_flag" in df.columns:
        rogue = df["tce_rogue_flag"].fillna(0) == 1
        df = df[~rogue]
        print(f"  dropped {rogue.sum():,} rogue TCEs")

    # Drop unvetted rows and map to binary.
    print("\nlabel counts before mapping:")
    print(df["av_training_set"].value_counts(dropna=False).to_string())
    df = df[df["av_training_set"].isin(LABEL_MAP)].copy()
    df["label"] = df["av_training_set"].map(LABEL_MAP).astype(int)

    kept = [c for c in CANDIDATE_FEATURES if c in df.columns]
    missing = [c for c in CANDIDATE_FEATURES if c not in df.columns]
    if missing:
        print(f"\n  not in this file, skipping: {', '.join(missing)}")

    keep_cols = sorted(set(PASSTHROUGH + kept + ["label"]))
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    print(f"\ncleaned: {n0:,} -> {len(df):,} rows, {df.kepid.nunique():,} stars")
    print(f"  {df.label.sum():,} planet candidates / {(1 - df.label).sum():,} false positives")
    print(f"  imbalance ratio: {(1 - df.label).sum() / df.label.sum():.2f} to 1")
    return df, kept


def engineer(df, base_features):
    """Derived features. Each one has a physical reason to exist."""
    f = list(base_features)
    eps = 1e-12

    # Heavily skewed quantities. Logs make them usable by linear models.
    for col, new in [
        ("tce_depth", "log_depth"),
        ("tce_model_snr", "log_snr"),
        ("tce_period", "log_period"),
    ]:
        if col in df.columns:
            df[new] = np.log10(df[col].clip(lower=eps))
            f.append(new)

    # Observed duration vs the duration expected for a central circular transit.
    # A real planet lands near 1. Eclipsing binaries and junk often do not.
    if {"tce_period", "tce_dor", "tce_duration"}.issubset(df.columns):
        expected_hrs = 24.0 * df["tce_period"] / (np.pi * df["tce_dor"].clip(lower=eps))
        df["duration_ratio"] = df["tce_duration"] / expected_hrs.clip(lower=eps)
        f.append("duration_ratio")

    # Fraction of the orbit spent in transit.
    if {"tce_duration", "tce_period"}.issubset(df.columns):
        df["duty_cycle"] = (df["tce_duration"] / 24.0) / df["tce_period"].clip(lower=eps)
        f.append("duty_cycle")

    # Steepness proxy. Deep-and-fast looks different from shallow-and-slow.
    if {"tce_depth", "tce_duration"}.issubset(df.columns):
        df["depth_per_hour"] = df["tce_depth"] / df["tce_duration"].clip(lower=eps)
        f.append("depth_per_hour")

    # Largest centroid offset. Big shift means the dip came from a nearby star.
    cen = [c for c in ["tce_dicco_msky", "tce_dikco_msky"] if c in df.columns]
    if cen:
        df["centroid_offset_max"] = df[cen].abs().max(axis=1)
        f.append("centroid_offset_max")

    print(f"\nengineered {len(f) - len(base_features)} new features, {len(f)} total")
    return df, f


def impute(df, features):
    """Median fill, with a missingness indicator where it is common.
    Columns that are entirely empty are dropped, not filled."""
    empty = [c for c in features if df[c].isna().all()]
    if empty:
        print(f"  dropped {len(empty)} fully-empty columns: {', '.join(empty)}")
        df = df.drop(columns=empty)
        features = [c for c in features if c not in empty]

    for c in features:
        n_missing = df[c].isna().sum()
        if n_missing == 0:
            continue
        pct = 100 * n_missing / len(df)
        if pct > 5:
            df[f"{c}_missing"] = df[c].isna().astype(int)
            features.append(f"{c}_missing")
        df[c] = df[c].fillna(df[c].median())
        print(f"  {c}: filled {n_missing:,} ({pct:.1f}%)")
    return df, features


def split(df, seed=42):
    """70/15/15 grouped by kepid, stratified on label."""
    y, g = df["label"].values, df["kepid"].values
    idx = np.arange(len(df))

    # 1/7 out for test (~14.3%)
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    rest_i, test_i = next(sgkf.split(idx, y, groups=g))

    # 1/6 of the remainder out for val (~14.3% of total)
    sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    tr_rel, val_rel = next(sgkf2.split(rest_i, y[rest_i], groups=g[rest_i]))
    train_i, val_i = rest_i[tr_rel], rest_i[val_rel]

    return df.iloc[train_i].copy(), df.iloc[val_i].copy(), df.iloc[test_i].copy()


def verify(train, val, test):
    """Hard assertion that no star appears in more than one split."""
    s = {"train": set(train.kepid), "val": set(val.kepid), "test": set(test.kepid)}
    ok = True
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = s[a] & s[b]
        print(f"  {a} vs {b}: {len(overlap)} shared stars")
        if overlap:
            ok = False
    assert ok, "LEAKAGE: a star appears in more than one split"

    total = len(train) + len(val) + len(test)
    print()
    for name, d in [("train", train), ("val", val), ("test", test)]:
        print(
            f"  {name:5s} {len(d):6,} rows ({100*len(d)/total:4.1f}%)  "
            f"{d.kepid.nunique():5,} stars  "
            f"{100*d.label.mean():4.1f}% candidates"
        )
    print("\n  LEAKAGE CHECK PASSED")


def class_chart(df, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = df["label"].value_counts().sort_index()
    labels = ["False positive\n(AFP + NTP)", "Planet candidate\n(PC)"]
    colors = ["#E8734A", "#3B7DD8"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, counts.values, color=colors, width=0.6)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({100*v/len(df):.1f}%)",
                ha="center", va="bottom", fontsize=10)

    ratio = counts[0] / counts[1]
    ax.set_ylabel("Number of TCEs")
    ax.set_title(f"DR24 class balance: {ratio:.1f} false positives per candidate")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(outdir, "class_balance.png")
    fig.savefig(path, dpi=150)
    print(f"\nwrote {path}")
    print(f"  all-negative baseline accuracy would be {100*counts[0]/len(df):.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--outdir", default="out")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df = load(a.csv)
    check_label(df)

    print("\n--- LEAKAGE GUARD ---")
    df = drop_leaky(df)

    print("\n--- CLEANING ---")
    df, base = clean(df)

    print("\n--- FEATURE ENGINEERING ---")
    df, features = engineer(df, base)

    print("\n--- MISSING VALUES ---")
    df, features = impute(df, features)

    print("\n--- SPLIT ---")
    train, val, test = split(df, a.seed)
    verify(train, val, test)

    for name, d in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(a.outdir, f"{name}.csv")
        d.to_csv(path, index=False)
        print(f"wrote {path}")

    with open(os.path.join(a.outdir, "feature_list.txt"), "w") as fh:
        fh.write("\n".join(features))
    print(f"wrote {os.path.join(a.outdir, 'feature_list.txt')} ({len(features)} features)")

    class_chart(df, a.outdir)


if __name__ == "__main__":
    main()