import pandas as pd

sheet_id = "15tEvxrNqecs6zzYh8Gc_WcgWGXI_9LHVtNZeUWR_skE"
sheet_names = ["Score_Filtered"]

dfs = []

for sheet in sheet_names:
    url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq"
        f"?tqx=out:csv"
        f"&sheet={sheet}"
        f"&range=A1:O1000"
    )

    df = pd.read_csv(url)

    # Ambil kolom berdasarkan POSISI
    df = df.iloc[:, [0, 1, 8, 9, 10, 12, 14]]

    # RESET NAMA KOLOM (INI KUNCI)
    df.columns = [
        "HomeTeam",
        "AwayTeam",
        "HomeForm",
        "AwayForm",
        "HomeOdds",
        "DrawOdds",
        "AwayOdds"
    ]

    # Optional: tandai asal sheet
    # df["source"] = sheet

    dfs.append(df)

# CONCAT KE BAWAH (ROW)
final_df = pd.concat(dfs, axis=0, ignore_index=True)

final_df.to_csv("footystats_combined.csv", index=False)

print("OK. Total rows:", len(final_df))
