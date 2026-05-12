import pickle

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'microwave':    10.0,
    'washer dryer':  0.5,
}

DISPLAY_NAMES = {
    'dish washer':  'Dishwasher',
    'fridge':       'Fridge',
    'microwave':    'Microwave',
    'washer dryer': 'Washing Machine',
}

for split in ['train', 'val', 'test']:
    with open(f'data/ukdale/{split}_small.pkl', 'rb') as f:
        raw = pickle.load(f)
    df = raw[0]

    print(f"\n{split.upper()} ({df.index[0].date()})")
    print(f"{'Appliance':<20} {'ON':>8} {'OFF':>8} {'TOTAL':>8}")
    print("-" * 48)
    for col in ['dish washer', 'fridge', 'microwave', 'washer dryer']:
        thr   = THRESHOLDS[col]
        on    = int((df[col] > thr).sum())
        total = len(df)
        off   = total - on
        name  = DISPLAY_NAMES[col]
        print(f"{name:<20} {on:>8} {off:>8} {total:>8}")
