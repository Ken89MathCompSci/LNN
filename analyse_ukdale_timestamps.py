import pickle
import numpy as np
import pandas as pd

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']
THRESHOLDS = {'dish washer': 10.0, 'fridge': 10.0, 'microwave': 10.0, 'washer dryer': 0.5}

def get_on_intervals(series, threshold):
    """Return list of (start, end, duration_min, max_W) for each ON block."""
    on = series > threshold
    intervals = []
    in_block = False
    start = None
    for ts, val in on.items():
        if val and not in_block:
            in_block = True
            start = ts
        elif not val and in_block:
            in_block = False
            seg = series[start:ts]
            dur = (ts - start).total_seconds() / 60
            intervals.append((start, ts, dur, seg.max()))
    if in_block:
        seg = series[start:]
        dur = (series.index[-1] - start).total_seconds() / 60
        intervals.append((start, series.index[-1], dur, seg.max()))
    return intervals

for split in ['train', 'val', 'test']:
    with open(f'data/ukdale/{split}_small.pkl', 'rb') as f:
        raw = pickle.load(f)
    df = raw[0] if isinstance(raw, (list, tuple)) else raw
    date = df.index[0].date()

    print(f"\n{'='*70}")
    print(f"  {split.upper()} split  ({date})")
    print(f"{'='*70}")

    for app in APPLIANCES:
        thr = THRESHOLDS[app]
        intervals = get_on_intervals(df[app], thr)
        total_on = df[app][df[app] > thr].shape[0] * 6 / 60  # minutes

        print(f"\n  {app}  (threshold={thr}W, {len(intervals)} ON-blocks, "
              f"{total_on:.1f} min total ON)")
        if len(intervals) == 0:
            print("    (never ON)")
        elif len(intervals) > 30:
            # Too many short bursts — show distribution instead
            durs = [iv[2] for iv in intervals]
            maxw = [iv[3] for iv in intervals]
            print(f"    Too many events ({len(intervals)}) to list individually.")
            print(f"    Duration  : min={min(durs):.1f}m  max={max(durs):.1f}m  "
                  f"mean={np.mean(durs):.1f}m  median={np.median(durs):.1f}m")
            print(f"    Peak (W)  : min={min(maxw):.0f}  max={max(maxw):.0f}  "
                  f"mean={np.mean(maxw):.0f}")
            print(f"    First 10 ON-blocks:")
            for iv in intervals[:10]:
                print(f"      {iv[0].strftime('%H:%M:%S')} - {iv[1].strftime('%H:%M:%S')}  "
                      f"({iv[2]:.1f} min, peak {iv[3]:.0f}W)")
            print(f"    Last 5 ON-blocks:")
            for iv in intervals[-5:]:
                print(f"      {iv[0].strftime('%H:%M:%S')} - {iv[1].strftime('%H:%M:%S')}  "
                      f"({iv[2]:.1f} min, peak {iv[3]:.0f}W)")
        else:
            for iv in intervals:
                print(f"    {iv[0].strftime('%H:%M:%S')} - {iv[1].strftime('%H:%M:%S')}  "
                      f"({iv[2]:.1f} min, peak {iv[3]:.0f}W)")
