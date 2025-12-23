import pandas as pd

sheet_id = "15tEvxrNqecs6zzYh8Gc_WcgWGXI_9LHVtNZeUWR_skE"
sheet_name = "Footystats_Data"   # sheet tujuan FILTER

url = (
    f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq"
    f"?tqx=out:csv"
    f"&sheet={sheet_name}"
    f"&range=A1:O1000"
)

df = pd.read_csv(url)
print(f'Extracted {len(df)} rows')
df.to_csv('footystats_gsheets.csv')