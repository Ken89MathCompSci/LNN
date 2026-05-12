import pickle
import numpy as np
import pandas as pd

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']

splits = {}
for split in ['train', 'val', 'test']:
    with open(f'data/ukdale/{split}_small.pkl', 'rb') as f:
        raw = pickle.load(f)
    df = raw[0] if isinstance(raw, (list, tuple)) else raw
    splits[split] = df

    step = df.index[1] - df.index[0] if len(df) > 1 else None
    print(f"\n{'='*60}")
    print(f"  {split}_small.pkl")
    print(f"{'='*60}")
    print(f"  Rows          : {len(df):,}")
    print(f"  Date range    : {df.index.min()} to {df.index.max()}")
    print(f"  Time step     : {step}")
    print(f"  Columns       : {list(df.columns)}")
    print()

    for col in df.columns:
        s = df[col]
        on_mask = s > (10.0 if col != 'washer dryer' else 0.5)
        pct_on  = on_mask.mean() * 100
        n_on    = on_mask.sum()
        print(f"  {col:<16}  min={s.min():8.2f}W  max={s.max():8.2f}W  "
              f"mean={s.mean():7.2f}W  std={s.std():7.2f}W  "
              f"ON={pct_on:5.1f}%  ON_rows={n_on:,}")

# Cross-split stats
print(f"\n{'='*60}")
print("  Cross-split ON% comparison (threshold: DW/FR/MW=10W, WD=0.5W)")
print(f"{'='*60}")
header = f"{'Appliance':<16}" + "".join(f"  {s:>8}" for s in splits)
print(header)
for col in ['main'] + APPLIANCES:
    row = f"{col:<16}"
    thr = 0.5 if col == 'washer dryer' else 10.0
    for split, df in splits.items():
        pct = (df[col] > thr).mean() * 100 if col in df.columns else float('nan')
        row += f"  {pct:7.1f}%"
    print(row)

# Event counts (appliance cycles)
print(f"\n{'='*60}")
print("  Approximate ON-cycles per split (consecutive ON blocks)")
print(f"{'='*60}")
for split, df in splits.items():
    print(f"\n  {split}:")
    for app in APPLIANCES:
        thr  = 0.5 if app == 'washer dryer' else 10.0
        on   = (df[app] > thr).astype(int)
        rises = (on.diff() == 1).sum()
        total_on_min = on.sum() * 6 / 60  # 6-second steps to minutes
        print(f"    {app:<16}: {rises:3d} ON-events, "
              f"{total_on_min:7.1f} min ON total")

# NaN check
print(f"\n{'='*60}")
print("  Missing values")
print(f"{'='*60}")
for split, df in splits.items():
    nans = df.isnull().sum()
    total = len(df)
    print(f"  {split}: {nans.to_dict()}  (total rows={total:,})")
