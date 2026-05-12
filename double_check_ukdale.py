import pickle
import numpy as np
import pandas as pd

for split in ['train', 'val', 'test']:
    with open(f'data/ukdale/{split}_small.pkl', 'rb') as f:
        raw = pickle.load(f)
    df = raw[0] if isinstance(raw, (list, tuple)) else raw

    mw = df['microwave']
    wd = df['washer dryer']

    print(f"\n{'='*60}")
    print(f"  {split.upper()}  ({df.index[0].date()})")
    print(f"{'='*60}")

    # Are raw values identical?
    identical_vals = (mw.values == wd.values).all()
    n_same = (mw.values == wd.values).sum()
    print(f"  Raw values identical (all rows)  : {identical_vals}")
    print(f"  Rows where MW == WD              : {n_same:,} / {len(df):,}  "
          f"({100*n_same/len(df):.1f}%)")

    # Correlation
    corr = mw.corr(wd)
    print(f"  Pearson correlation(MW, WD)      : {corr:.6f}")

    # Value ranges side-by-side
    print(f"\n  MW  min={mw.min():.2f}  max={mw.max():.2f}  "
          f"mean={mw.mean():.2f}  std={mw.std():.2f}")
    print(f"  WD  min={wd.min():.2f}  max={wd.max():.2f}  "
          f"mean={wd.mean():.2f}  std={wd.std():.2f}")

    # ON/OFF pattern comparison
    mw_on = (mw > 10.0)
    wd_on = (wd > 0.5)
    both_on   = (mw_on & wd_on).sum()
    mw_only   = (mw_on & ~wd_on).sum()
    wd_only   = (~mw_on & wd_on).sum()
    both_off  = (~mw_on & ~wd_on).sum()
    print(f"\n  ON/OFF pattern (MW thr=10W, WD thr=0.5W):")
    print(f"    Both ON        : {both_on:,}")
    print(f"    MW only ON     : {mw_only:,}")
    print(f"    WD only ON     : {wd_only:,}")
    print(f"    Both OFF       : {both_off:,}")

    # Show first 20 rows where they differ in raw value
    diff_mask = mw.values != wd.values
    diff_idx  = np.where(diff_mask)[0]
    if len(diff_idx) == 0:
        print(f"\n  No rows differ in raw value.")
    else:
        print(f"\n  First 20 rows where MW != WD (raw):")
        print(f"    {'Timestamp':<22}  {'MW (W)':>10}  {'WD (W)':>10}  {'diff':>10}")
        for idx in diff_idx[:20]:
            ts  = df.index[idx]
            mvv = mw.iloc[idx]
            wvv = wd.iloc[idx]
            print(f"    {str(ts):<22}  {mvv:>10.2f}  {wvv:>10.2f}  {mvv-wvv:>10.2f}")

    # Sample of actual values during "ON" periods
    print(f"\n  Sample MW values when MW > 10W (first 10):")
    mw_on_vals = mw[mw > 10].head(10)
    for ts, v in mw_on_vals.items():
        print(f"    {ts}  {v:.2f}W")

    print(f"\n  Sample WD values when WD > 0.5W (first 10):")
    wd_on_vals = wd[wd > 0.5].head(10)
    for ts, v in wd_on_vals.items():
        print(f"    {ts}  {v:.2f}W")
