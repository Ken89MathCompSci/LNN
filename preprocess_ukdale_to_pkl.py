"""
UK-DALE → Test PKL
===================
Prepares a test pkl file from UK-DALE HDF5 so that already-trained
models (trained on REDD) can be evaluated on UK-DALE appliances.

Output: data/ukdale/test_small.pkl
        Same format as data/redd/test_small.pkl:
            pickle.load(f)[0]  →  pandas DataFrame
            columns: ['main', 'kettle', 'toaster', ...]

Usage:
    py preprocess_ukdale_to_pkl.py

To see all available appliances first:
    Set PRINT_METERS = True below, then run.
"""

import os
import pickle
import tables
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

H5_PATH  = 'ukdale/ukdale.h5'
BUILDING = 1
OUT_PATH = 'data/ukdale/test_small.pkl'

# Resample to 1-minute intervals (matches REDD pkl resolution)
RESAMPLE_FREQ = '1T'

# Forward-fill gaps up to 60 minutes (60 steps at 1-min resolution)
FFILL_LIMIT = 60

# Test window — UK-DALE House 1 has data from 2012-11-09 to 2015-01-05
TEST_START = '2014-01-01'
TEST_END   = '2014-06-30'

# Appliances to include → column name : meter ID
# House 1 meter map (set PRINT_METERS=True to see all):
#   5=washer dryer  6=dish washer  10=kettle  11=toaster
#   12=fridge       13=microwave   36=coffee maker
#   39=hair dryer   41=clothes iron  42=oven
APPLIANCE_METERS = {
    'kettle':       10,
    'toaster':      11,
    'fridge':       12,
    'microwave':    13,
    'dish washer':   6,
    'washer dryer':  5,
}

# Set True to print all meter→appliance names then exit
PRINT_METERS = False

# ══════════════════════════════════════════════════════════════════════════════


def print_available_meters():
    try:
        from nilmtk import DataSet
        ds = DataSet(H5_PATH)
        elec = ds.buildings[BUILDING].elec
        print(f"\nBuilding {BUILDING} appliances:")
        print(f"  {'Meter':<10} Appliance")
        print(f"  {'-'*35}")
        for meter in elec.meters:
            try:
                name = meter.appliances[0].type['type'] if meter.appliances else 'mains'
                print(f"  meter{meter.instance():<6} {name}")
            except Exception:
                print(f"  meter{meter.instance():<6} unknown")
    except ImportError:
        print("nilmtk not installed — cannot resolve appliance names")


def load_meter(meter_id):
    """Load one meter, resample, forward-fill → pd.Series"""
    with tables.open_file(H5_PATH, 'r') as h5:
        b  = getattr(h5.root, f'building{BUILDING}')
        m  = getattr(b.elec, f'meter{meter_id}')
        raw = m.table[:]

    idx = pd.to_datetime(raw['index'], unit='ns', utc=True)
    val = raw['values_block_0'].flatten().astype(np.float32)

    s = pd.Series(val, index=idx)
    s = s[~s.index.duplicated(keep='last')].sort_index()
    s = s.clip(lower=0)
    s = s.resample(RESAMPLE_FREQ).mean()
    s = s.ffill(limit=FFILL_LIMIT)
    return s


def main():
    if not os.path.exists(H5_PATH):
        print(f"ERROR: {H5_PATH} not found.")
        return

    if PRINT_METERS:
        print_available_meters()
        return

    print(f"Loading UK-DALE Building {BUILDING}  [{TEST_START} → {TEST_END}]")
    print(f"Resample: {RESAMPLE_FREQ}  ffill limit: {FFILL_LIMIT} steps\n")

    # Load all meters
    print("  Loading aggregate (meter1)...")
    main_s = load_meter(1)
    main_s.name = 'main'
    print(f"    {len(main_s):,} rows")

    series_list = [main_s]
    for name, mid in APPLIANCE_METERS.items():
        print(f"  Loading {name} (meter{mid})...")
        try:
            s = load_meter(mid)
            s.name = name
            print(f"    {len(s):,} rows")
            series_list.append(s)
        except Exception as e:
            print(f"    SKIP — {e}")

    # Merge on shared timestamps
    df = pd.concat(series_list, axis=1, join='inner').dropna()
    print(f"\nMerged: {df.shape}  ({df.index[0].date()} → {df.index[-1].date()})")

    # Slice test window
    test_df = df.loc[TEST_START:TEST_END].copy()

    if len(test_df) == 0:
        print(f"\nERROR: No data in {TEST_START}→{TEST_END}. "
              f"Data covers {df.index[0].date()}→{df.index[-1].date()}")
        return

    print(f"Test slice: {len(test_df):,} rows  "
          f"({test_df.index[0].date()} → {test_df.index[-1].date()})")

    # Print on/off stats
    THRESHOLDS = {
        'kettle': 10.0, 'toaster': 10.0, 'fridge': 10.0,
        'microwave': 10.0, 'dish washer': 10.0, 'washer dryer': 0.5,
        'coffee maker': 10.0, 'hair dryer': 10.0,
        'clothes iron': 10.0, 'oven': 10.0,
    }
    print(f"\n  {'Appliance':<20} {'ON':>8} {'OFF':>8} {'TOTAL':>8}")
    print(f"  {'-'*48}")
    for col in test_df.columns:
        if col == 'main':
            continue
        thr = THRESHOLDS.get(col, 10.0)
        on  = int((test_df[col] > thr).sum())
        off = int((test_df[col] <= thr).sum())
        print(f"  {col:<20} {on:>8,} {off:>8,} {len(test_df):>8,}")

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump([test_df], f)
    print(f"\nSaved → {OUT_PATH}")
    print(f"Columns: {list(test_df.columns)}")
    print("\nLoad it in your test script with:")
    print(f"    with open('{OUT_PATH}', 'rb') as f:")
    print(f"        test_data = pickle.load(f)[0]")


if __name__ == '__main__':
    main()
