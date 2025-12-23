#!/usr/bin/env python3
"""
predict_bet.py

Usage:
  # batch from CSV:
  python predict_bet.py --csv input_matches.csv --out output_with_decisions.csv

Input CSV should contain at least: HomeForm, AwayForm, HomeOdds, DrawOdds, AwayOdds
Optional but recommended: HomeTeam, AwayTeam, SourceDate

Output:
  CSV with appended columns including HomeTeam and AwayTeam (if present),
  plus FormDiff, FormGroup, Prob_*, EV_*, BestSide, BestEV, Decision, stakes, ProbMargin.
"""
import joblib, json
import numpy as np
import pandas as pd
import argparse, os, re, sys

ART_PREFIX = "pipeline_artifact"
ART_DIR = "pipeline_footystats"  # adjust if your artifacts are elsewhere

IMP_PATH = f"{ART_PREFIX}_imputer.joblib"
CALIB_PATH = f"{ART_PREFIX}_calib_model.joblib"
KMEANS_PATH = f"{ART_PREFIX}_kmeans_cuts.json"
CONF_PATH = f"{ART_PREFIX}_config.json"

# ----------------- helper: numeric extraction -----------------
num_re = re.compile(r"(-?\d+[.,]\d+|-?\d+)")

def extract_first_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = num_re.search(s)
    if not m:
        return np.nan
    tok = m.group(0).replace(',', '.')
    try:
        return float(tok)
    except:
        return np.nan

# ----------------- load artifacts -----------------
if not os.path.exists(IMP_PATH) or not os.path.exists(CALIB_PATH):
    print("ERROR: required artifact files not found. Expected:")
    print(" ", IMP_PATH)
    print(" ", CALIB_PATH)
    print("Place your artifacts in folder:", ART_DIR)
    sys.exit(1)

imp = joblib.load(IMP_PATH)
calib = joblib.load(CALIB_PATH)

cuts = {}
if os.path.exists(KMEANS_PATH):
    with open(KMEANS_PATH) as f:
        cuts = json.load(f)
else:
    cuts = {'cut1': -0.340, 'cut2': 0.714}
with open(CONF_PATH) as f:
    config = json.load(f)

cut1 = cuts.get('cut1', -0.340)
cut2 = cuts.get('cut2', 0.714)
EV_THRESH = config.get('EV_THRESH', 0.05)
PROB_MARGIN = config.get('PROB_MARGIN', 0.05)
HOME_MIN = config.get('HOME_ODDS_MIN', 1.50)
HOME_MAX = config.get('HOME_ODDS_MAX', 2.40)
AWAY_MIN = config.get('AWAY_ODDS_MIN', 2.20)
AWAY_MAX = config.get('AWAY_ODDS_MAX', 4.00)
KELLY_F = config.get('KELLY_F', 0.25)

# ----------------- decision logic -----------------
def kelly_fraction(p, odds, f=0.25):
    b = odds - 1.0
    if b <= 0:
        return 0.0
    k = (p*b - (1-p)) / b
    return max(0, k) * f

def decide_single(home_form, away_form, home_odds, draw_odds, away_odds):
    # prepare features
    formdiff = float(home_form) - float(away_form)
    imp_home = 1.0/float(home_odds)
    imp_draw = 1.0/float(draw_odds)
    imp_away = 1.0/float(away_odds)
    s = imp_home + imp_draw + imp_away
    imp_home_n = imp_home / s
    imp_draw_n = imp_draw / s
    imp_away_n = imp_away / s
    feat = np.array([[formdiff, imp_home_n, imp_draw_n, imp_away_n, 1]])
    feat_imp = imp.transform(feat)  # use fitted imputer
    probs = calib.predict_proba(feat_imp)[0]  # order corresponds to calib.classes_
    classes = list(calib.classes_)
    # map to keys
    prob_map = {'HomeWin':0.0, 'Draw':0.0, 'AwayWin':0.0}
    for i,cls in enumerate(classes):
        prob_map[cls] = probs[i]
    p_home = prob_map['HomeWin']; p_draw = prob_map['Draw']; p_away = prob_map['AwayWin']
    # EVs
    ev_home = p_home * float(home_odds) - 1.0
    ev_draw = p_draw * float(draw_odds) - 1.0
    ev_away = p_away * float(away_odds) - 1.0
    evs = {'Home': ev_home, 'Draw': ev_draw, 'Away': ev_away}
    best_side = max(evs, key=evs.get)
    best_ev = evs[best_side]
    # form group check
    if formdiff < cut1:
        form_group = 'Away Strong'
    elif formdiff > cut2:
        form_group = 'Home Strong'
    else:
        form_group = 'Balanced'
    # odds range check
    ok_odds = False
    if best_side == 'Home':
        ok_odds = HOME_MIN <= home_odds <= HOME_MAX
        p_model, p_market = p_home, imp_home_n
    elif best_side == 'Away':
        ok_odds = AWAY_MIN <= away_odds <= AWAY_MAX
        p_model, p_market = p_away, imp_away_n
    else:
        ok_odds = True
        p_model, p_market = p_draw, imp_draw_n

    # decision: require all
    decision = (best_ev >= EV_THRESH) and (p_model >= p_market + PROB_MARGIN) and (form_group in ('Home Strong','Away Strong')) and ok_odds
    # stake suggestions
    stake_flat = 1.0 if decision else 0.0
    stake_kelly_unit = kelly_fraction(p_home if best_side=='Home' else (p_away if best_side=='Away' else p_draw),
                                      home_odds if best_side=='Home' else (away_odds if best_side=='Away' else draw_odds),
                                      f=KELLY_F) if decision else 0.0

    return {
        'FormDiff': formdiff,
        'FormGroup': form_group,
        'Prob_Home': p_home, 'Prob_Draw': p_draw, 'Prob_Away': p_away,
        'EV_Home': ev_home, 'EV_Draw': ev_draw, 'EV_Away': ev_away,
        'BestSide': best_side, 'BestEV': best_ev,
        'MarketImpHome': imp_home_n, 'MarketImpAway': imp_away_n, 'MarketImpDraw': imp_draw_n,
        'OddsHome': home_odds, 'OddsDraw': draw_odds, 'OddsAway': away_odds,
        'Decision': 'BET' if decision else 'NO_BET',
        'StakeFlat': stake_flat, 'StakeKellyUnit': stake_kelly_unit,
        'ProbMargin': (p_model - p_market) if 'p_model' in locals() else np.nan
    }

# ----------------- batch processing -----------------
def process_csv(infile, outfile=None, preview=5):
    df = pd.read_csv(infile)
    # normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]

    # keep team columns if present
    keep_team_cols = []
    for t in ['HomeTeam','AwayTeam','SourceDate']:
        if t in df.columns:
            keep_team_cols.append(t)

    # try to extract numeric from possibly dirty strings
    for c in ['HomeForm','AwayForm','HomeOdds','DrawOdds','AwayOdds']:
        if c in df.columns:
            df[c+'_parsed'] = df[c].apply(extract_first_number)
        else:
            df[c+'_parsed'] = np.nan

    # If original numeric present, prefer that
    for c in ['HomeForm','AwayForm','HomeOdds','DrawOdds','AwayOdds']:
        parsed = c + '_parsed'
        if c in df.columns:
            # coerce original to numeric where possible
            df[c+'_num'] = pd.to_numeric(df[c], errors='coerce')
            # fill na in numeric with parsed
            df[c+'_num'] = df[c+'_num'].fillna(df[parsed])
        else:
            df[c+'_num'] = df[parsed]

    # Drop rows that still have missing critical fields
    req_num = ['HomeForm_num','AwayForm_num','HomeOdds_num','DrawOdds_num','AwayOdds_num']
    missing_mask = df[req_num].isna().any(axis=1)
    if missing_mask.all():
        raise ValueError("No valid rows with required numeric columns after parsing.")
    df_valid = df[~missing_mask].copy().reset_index(drop=True)
    print(f"Rows in: {len(df)}, valid after parsing: {len(df_valid)}")

    # iterate and decide
    results = []
    for idx, row in df_valid.iterrows():
        try:
            res = decide_single(
                float(row['HomeForm_num']),
                float(row['AwayForm_num']),
                float(row['HomeOdds_num']),
                float(row['DrawOdds_num']),
                float(row['AwayOdds_num'])
            )
            # build output row, include team info if present
            out = {}
            for t in keep_team_cols:
                out[t] = row.get(t, None)
            # original raw fields for traceability
            out.update({
                'HomeForm_raw': row.get('HomeForm', None),
                'AwayForm_raw': row.get('AwayForm', None),
                'HomeOdds_raw': row.get('HomeOdds', None),
                'DrawOdds_raw': row.get('DrawOdds', None),
                'AwayOdds_raw': row.get('AwayOdds', None),
                'HomeForm': row['HomeForm_num'],
                'AwayForm': row['AwayForm_num'],
                'HomeOdds': row['HomeOdds_num'],
                'DrawOdds': row['DrawOdds_num'],
                'AwayOdds': row['AwayOdds_num'],
            })
            out.update(res)
            results.append(out)
        except Exception as e:
            print(f"Row {idx} processing error: {e}")

    if not results:
        print("No results to write.")
        return None

    df_out = pd.DataFrame(results)
    if outfile is None:
        base = os.path.splitext(os.path.basename(infile))[0]
        outfile = f"{base}_predictions.csv"
    df_out.to_csv(outfile, index=False)
    print(f"Wrote {len(df_out)} rows to {outfile}")
    print(df_out[[c for c in ['HomeTeam','AwayTeam','FormDiff','FormGroup','BestSide','BestEV','Decision'] if c in df_out.columns]].head(preview).to_string(index=False))
    return outfile

# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser(description="Predict bets from CSV or run demo single example")
    parser.add_argument('--csv', help='Input CSV file path for batch processing')
    parser.add_argument('--out', help='Output CSV file path (optional)')
    args = parser.parse_args()

    if args.csv:
        process_csv(args.csv, outfile=args.out)
    else:
        # demo single run
        h_form = float(input("Home Form? "))
        a_form = float(input("Away Form? "))
        h_odds = float(input("Home Odds? "))
        d_odds = float(input("Draw Odds? "))
        a_odds = float(input("Away Odds? "))
        example = decide_single(home_form=h_form, away_form=a_form, home_odds=h_odds, draw_odds=d_odds, away_odds=a_odds)
        import pprint; pprint.pprint(example)

if __name__ == "__main__":
    main()
