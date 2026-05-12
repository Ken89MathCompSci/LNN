import pickle

for split in ['train', 'val', 'test']:
    with open(f'data/ukdale/{split}_small.pkl', 'rb') as f:
        raw = pickle.load(f)

    print(f"\n=== {split}_small.pkl ===")
    print(f"  type(raw)  : {type(raw)}")
    if isinstance(raw, (list, tuple)):
        print(f"  len(raw)   : {len(raw)}")
        for j, item in enumerate(raw):
            print(f"  --- raw[{j}] : {type(item)} ---")
            if hasattr(item, 'columns'):
                print(f"    shape      : {item.shape}")
                print(f"    time start : {item.index.min()}")
                print(f"    time end   : {item.index.max()}")
                print(f"    data points: {len(item)}")
                print(f"    columns    : {list(item.columns)}")
                # Check for house info in index name or attrs
                print(f"    index.name : {item.index.name}")
                print(f"    attrs      : {item.attrs if hasattr(item, 'attrs') else 'N/A'}")
            elif isinstance(item, dict):
                print(f"    keys: {list(item.keys())}")
            else:
                print(f"    value: {item}")
    elif hasattr(raw, 'columns'):
        print(f"  shape      : {raw.shape}")
        print(f"  time start : {raw.index.min()}")
        print(f"  time end   : {raw.index.max()}")
        print(f"  data points: {len(raw)}")
        print(f"  index.name : {raw.index.name}")
        print(f"  attrs      : {raw.attrs if hasattr(raw, 'attrs') else 'N/A'}")
