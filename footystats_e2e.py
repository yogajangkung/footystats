#!/usr/bin/env python3

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta

# ======================
# CONFIG
# ======================
SHEET_ID = "15tEvxrNqecs6zzYh8Gc_WcgWGXI_9LHVtNZeUWR_skE"
SHEET_NAME = "Footystats_Data"
HISTORY_CSV = "footystats_history.csv"
SNAPSHOT_CSV = "footystats_snapshot.csv"
OUTPUT_CSV = "footystats_predictions.csv"

# HOURS_BACK = 6
# MIN_SNAPSHOTS = 3
# MAX_ODDS_STD = 0.05

HOURS_BACK = 12
MIN_SNAPSHOTS = 2
MAX_ODDS_STD = 0.10

# model artifacts
ART_PREFIX = "pipeline_artifact"
IMP = joblib.load(f"{ART_PREFIX}_imputer.joblib")
CALIB = joblib.load(f"{ART_PREFIX}_calib_model.joblib")

with open(f"{ART_PREFIX}_config.json") as f:
    CFG = json.load(f)

# ======================
# STEP 1 — DOWNLOAD GSHEETS (APPEND)
# ======================
print("📥 Downloading Google Sheets...")

url = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq"
    f"?tqx=out:csv&sheet={SHEET_NAME}&range=A1:O1000"
)

df = pd.read_csv(url)
df["snapshot_time"] = pd.Timestamp.utcnow()

if os.path.exists(HISTORY_CSV):
    df.to_csv(HISTORY_CSV, mode="a", header=False, index=False)
else:
    df.to_csv(HISTORY_CSV, index=False)

print(f"✔ Snapshot appended: {len(df)} rows")

# ======================
# STEP 2 — SNAPSHOT FILTER (ODDS STABLE)
# ======================
print("🧮 Building stable odds snapshot...")

hist = pd.read_csv(HISTORY_CSV, parse_dates=["snapshot_time"])
cutoff = hist["snapshot_time"].max() - timedelta(hours=HOURS_BACK)
hist = hist[hist["snapshot_time"] >= cutoff]

rows = []

for (home, away), g in hist.groupby(["HomeTeam", "AwayTeam"]):
    if len(g) < MIN_SNAPSHOTS:
        continue

    odds_std = g["HomeOdds"].std()

    if odds_std > MAX_ODDS_STD:
        continue

    rows.append({
        "HomeTeam": home,
        "AwayTeam": away,
        "HomeForm": g["HomeForm"].iloc[-1],
        "AwayForm": g["AwayForm"].iloc[-1],
        "HomeOdds": g["HomeOdds"].median(),
        "DrawOdds": g["DrawOdds"].median(),
        "AwayOdds": g["AwayOdds"].median(),
        "SnapshotCount": len(g),
        "OddsStd": odds_std
    })

snap = pd.DataFrame(rows)
if snap.empty:
    print("⚠️ No stable matches yet — need more snapshots.")
    snap.to_csv(SNAPSHOT_CSV, index=False)
    exit(0)
snap.to_csv(SNAPSHOT_CSV, index=False)

print(f"✔ Stable matches: {len(snap)}")

# ======================
# STEP 3 — PREDICTION
# ======================
print("🤖 Running prediction model...")

def decide(row):
    h, a = row.HomeForm, row.AwayForm
    ho, do, ao = row.HomeOdds, row.DrawOdds, row.AwayOdds

    formdiff = h - a

    imp = np.array([1/ho, 1/do, 1/ao])
    imp /= imp.sum()

    X = np.array([[formdiff, imp[0], imp[1], imp[2], 1]])
    X = IMP.transform(X)

    probs = CALIB.predict_proba(X)[0]
    classes = CALIB.classes_

    p = dict(zip(classes, probs))
    p_home = p["HomeWin"]

    ev_home = p_home * ho - 1

    decision = (
        p_home >= 0.90 and
        ev_home >= CFG["EV_THRESH"]
    )

    return pd.Series({
        "Prob_Home": p_home,
        "EV_Home": ev_home,
        "Decision": "BET" if decision else "NO_BET"
    })

pred = snap.join(snap.apply(decide, axis=1))
pred["OddsSource"] = f"MEDIAN_{HOURS_BACK}H"

pred.to_csv(OUTPUT_CSV, index=False)

print("✅ DONE")
print(pred[[
    "HomeTeam", "AwayTeam",
    "Prob_Home", "EV_Home",
    "SnapshotCount", "OddsStd",
    "Decision"
]].sort_values("Prob_Home", ascending=False).head(10))
