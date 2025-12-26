import pandas as pd
import os
import requests
from datetime import datetime

# ==============================
# CONFIG
# ==============================
sheet_id = "15tEvxrNqecs6zzYh8Gc_WcgWGXI_9LHVtNZeUWR_skE"
sheet_names = ["Footystats_Data_Home", "Footystats_Data_Away"]

OUTPUT_CSV = "footystats_combined.csv"
SENT_FILE = "sent_telegram.csv"

TOKEN = "5639278356:AAFJf7TkrQna2yumFuEuUjFj1dvwgtLBq6g"
CHAT_ID = "434215579"


# ==============================
# SENT FIXTURE STORAGE
# ==============================
def load_sent_ids():
    if not os.path.exists(SENT_FILE):
        return set()
    df = pd.read_csv(SENT_FILE)
    return set(df["fixture_id"].astype(str))


def save_sent_ids(ids):
    if not ids:
        return

    df_new = pd.DataFrame({"fixture_id": list(ids)})

    if os.path.exists(SENT_FILE):
        df_old = pd.read_csv(SENT_FILE)
        df_all = pd.concat([df_old, df_new]).drop_duplicates()
    else:
        df_all = df_new

    df_all.to_csv(SENT_FILE, index=False)


# ==============================
# FIXTURE ID
# ==============================
def build_fixture_id(df):
    return (
        df["HomeTeam"].str.lower().str.strip()
        + "_vs_"
        + df["AwayTeam"].str.lower().str.strip()
    )


# ==============================
# TELEGRAM
# ==============================
def send_telegram(df):
    if df.empty:
        print("ℹ️ No data to send")
        return

    sent_ids = load_sent_ids()

    df = df.copy()
    df["fixture_id"] = build_fixture_id(df)

    # FILTER YANG BELUM PERNAH DIKIRIM
    df_new = df[~df["fixture_id"].isin(sent_ids)]

    if df_new.empty:
        print("ℹ️ No new fixtures")
        return

    lines = ["🔥 *FOOTYSTATS BET SIGNAL* 🔥\n"]

    for _, r in df_new.iterrows():
        line = (
            f"🏟️ *{r.HomeTeam} vs {r.AwayTeam}*\n"
            f"Prob Home: {r.Prob_Home:.2%}\n"
            f"Odds: {r.HomeOdds:.2f} | {r.DrawOdds:.2f} | {r.AwayOdds:.2f}\n"
            f"EV Home: {r.EV_Home:.2%}\n"
            "────────────"
        )
        lines.append(line)

    message = "\n".join(lines)

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    r = requests.post(url, data=payload, timeout=10)

    if r.ok and r.json().get("ok"):
        print(f"📨 Sent {len(df_new)} fixtures")
        save_sent_ids(set(df_new["fixture_id"]))
    else:
        print("❌ Telegram error:", r.text)


# ==============================
# MAIN PIPELINE
# ==============================
dfs = []

for sheet in sheet_names:
    url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq"
        f"?tqx=out:csv"
        f"&sheet={sheet}"
        f"&range=A1:O1000"
    )

    df = pd.read_csv(url, header=None)

    # Ambil kolom A,B,I,J,K,M,O (by index)
    df = df.iloc[:, [0, 1, 8, 9, 10, 12, 14]]

    # Rename kolom agar konsisten
    df.columns = [
        "HomeTeam",   # A
        "AwayTeam",   # B
        "Prob_Home",  # I
        "Prob_Away",  # J
        "HomeOdds",   # K
        "DrawOdds",   # M
        "AwayOdds"    # O
    ]

    df["EV_Home"] = (df["Prob_Home"] * df["HomeOdds"]) - 1
    df["source"] = sheet
    df["fetched_at"] = datetime.utcnow()

    dfs.append(df)
    print(sheet, len(df))

# GABUNG KE BAWAH
final_df = pd.concat(dfs, ignore_index=True)

# SIMPAN CSV
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ CSV saved: {OUTPUT_CSV} ({len(final_df)} rows)")

# KIRIM TELEGRAM (ANTI DUPLIKASI)
send_telegram(final_df)
