"""
Preprocess UKDALE data into train/val/test pkl files matching REDD format.

Format:
    list([DataFrame]) where DataFrame has columns:
    main, dish washer, fridge, microwave, washer dryer
    with datetime index, 6s resolution, 14400 rows = 24h

Splits:
    Train : House 1, first 24h  (~2013-03-17)
    Val   : House 1, 1 month later (~2013-04-17)
    Test  : House 5, first 24h  (~2014-06-29)
"""

import os
import pickle
import pandas as pd

# Paths
HOUSE1_CSV = 'c:/Users/MathK/OneDrive/Desktop/PhD_Slides/dataset_python/ukdale/UKDALE_House1_Final_6s.csv'
HOUSE5_CSV = 'c:/Users/MathK/OneDrive/Desktop/PhD_Slides/Processed_Data/UKDALE_House5_Processed.csv'
OUT_DIR    = 'data/ukdale'

ROWS_24H   = 14400  # 24h at 6s resolution

# Column mapping to match REDD format
HOUSE1_COL_MAP = {
    'Total_Power' : 'main',
    'Dishwasher'  : 'dish washer',
    'Fridge_Freezer': 'fridge',
    'Microwave'   : 'microwave',
    'Washer_Dryer': 'washer dryer',
}

HOUSE5_COL_MAP = {
    'Total_Power' : 'main',
    'Dishwasher'  : 'dish washer',
    'Fridge_Freezer': 'fridge',
    'Microwave'   : 'microwave',
    'Washer_Dryer': 'washer dryer',
}

TARGET_COLS = ['main', 'dish washer', 'fridge', 'microwave', 'washer dryer']


def find_row_for_date(csv_path, target_date, time_col='time', chunksize=50000):
    """Scan csv in chunks to find the first row index on or after target_date."""
    target = pd.Timestamp(target_date)
    row_idx = 0
    for chunk in pd.read_csv(csv_path, usecols=[time_col], chunksize=chunksize):
        chunk[time_col] = pd.to_datetime(chunk[time_col], infer_datetime_format=True)
        mask = chunk[time_col] >= target
        if mask.any():
            return row_idx + mask.idxmax() - chunk.index[0]
        row_idx += len(chunk)
    return row_idx


def load_window(csv_path, col_map, skiprows, time_col='time'):
    """Load exactly ROWS_24H rows starting from skiprows."""
    print(f"  Reading {ROWS_24H} rows from row {skiprows}...")
    # skiprows=range(1, skiprows+1) skips data rows but keeps header
    skip = range(1, skiprows + 1) if skiprows > 0 else None
    df = pd.read_csv(csv_path, skiprows=skip, nrows=ROWS_24H)
    df[time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True)
    df = df.rename(columns=col_map)
    df = df.set_index(time_col)
    df = df[TARGET_COLS]
    df = df.fillna(0.0)
    print(f"  Window: {df.shape}, {df.index.min()} to {df.index.max()}")
    return df


def save_pkl(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump([df], f)
    print(f"  Saved to {path}")


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    # Train: first 24h of House 1
    print("\nExtracting TRAIN (House 1, first 24h)...")
    train_df = load_window(HOUSE1_CSV, HOUSE1_COL_MAP, skiprows=0)
    save_pkl(train_df, f'{OUT_DIR}/train_small.pkl')

    # Val: House 1 starting ~1 month after train end
    print("\nFinding VAL start row (House 1, ~2013-04-17)...")
    val_start_idx = find_row_for_date(HOUSE1_CSV, '2013-04-17')
    print(f"  Val start row: {val_start_idx}")
    print("\nExtracting VAL...")
    val_df = load_window(HOUSE1_CSV, HOUSE1_COL_MAP, skiprows=val_start_idx)
    save_pkl(val_df, f'{OUT_DIR}/val_small.pkl')

    # Test: first 24h of House 5
    print("\nExtracting TEST (House 5, first 24h)...")
    test_df = load_window(HOUSE5_CSV, HOUSE5_COL_MAP, skiprows=0)
    save_pkl(test_df, f'{OUT_DIR}/test_small.pkl')

    # --- Verify ---
    print("\n=== Verification ===")
    for split, path in [('train', f'{OUT_DIR}/train_small.pkl'),
                        ('val',   f'{OUT_DIR}/val_small.pkl'),
                        ('test',  f'{OUT_DIR}/test_small.pkl')]:
        with open(path, 'rb') as f:
            df = pickle.load(f)[0]
        print(f"{split}: shape={df.shape}, cols={df.columns.tolist()}")
        print(f"  date range: {df.index.min()} to {df.index.max()}")
        print(f"  main range: {df['main'].min():.1f} to {df['main'].max():.1f} W")
        print()

    print("Done. UKDALE pkl files ready in data/ukdale/")
