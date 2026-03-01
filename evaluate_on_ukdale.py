"""
Evaluate ALL Models on UK-DALE Test Set
=========================================
Trains on REDD (train + val) → Evaluates on UK-DALE test pkl.

Appliance handling:
  Direct     : appliance exists in both REDD and UK-DALE
               → trained on REDD, tested on UK-DALE  (cross-dataset)
  Zero-shot  : appliance only in UK-DALE (kettle, toaster)
               → model trained on closest REDD proxy (microwave),
                 evaluated against actual UK-DALE ground truth

Run one model per Colab session to avoid timeouts.

Models (11 total):
  Baseline : gru, lstm, resnet, tcn, transformer
  LNN      : standard_lnn, advanced_lnn, attention_lnn,
             cnn_encoder, transformer_lnn, bidirectional_lnn

Step 1 — prepare test file (once):
    !python preprocess_ukdale_to_pkl.py

Step 2 — run one model per session:
    !python evaluate_on_ukdale.py --model gru
    !python evaluate_on_ukdale.py --model lstm
    !python evaluate_on_ukdale.py --model resnet
    !python evaluate_on_ukdale.py --model tcn
    !python evaluate_on_ukdale.py --model transformer
    !python evaluate_on_ukdale.py --model standard_lnn
    !python evaluate_on_ukdale.py --model advanced_lnn
    !python evaluate_on_ukdale.py --model attention_lnn
    !python evaluate_on_ukdale.py --model cnn_encoder
    !python evaluate_on_ukdale.py --model transformer_lnn
    !python evaluate_on_ukdale.py --model bidirectional_lnn

Step 3 — generate graphs after all models done:
    !python evaluate_on_ukdale.py --plot

Run all at once (if session allows):
    !python evaluate_on_ukdale.py --model all
"""

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.stats import truncnorm

sys.path.append('Source Code')

from models import (
    # Baseline
    GRUModel, LSTMModel, ResNetModel, TCNModel, SimpleTransformerModel,
    # LNN
    LiquidNetworkModel, AdvancedLiquidNetworkModel,
    AttentionLiquidNetworkModel, CNNEncoderLiquidNetworkModel,
    TransformerEncoderLiquidNetworkModel, BidirectionalEncoderLiquidNetworkModel,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────

REDD_TRAIN_PKL  = 'data/redd/train_small.pkl'
REDD_VAL_PKL    = 'data/redd/val_small.pkl'
UKDALE_TEST_PKL = 'data/ukdale/test_small.pkl'

EPOCHS   = 80
PATIENCE = 20
WIN      = 100

# Appliances present in BOTH REDD and UK-DALE test pkl
# The script auto-detects the intersection — this is a fallback filter
APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'microwave':    10.0,
    'washer dryer':  0.5,
    'kettle':       10.0,
    'toaster':      10.0,
    'coffee maker': 10.0,
    'hair dryer':   10.0,
    'clothes iron': 10.0,
    'oven':         10.0,
}

APP_LABELS = {
    'dish washer':  'Dish Washer',
    'fridge':       'Fridge',
    'microwave':    'Microwave',
    'washer dryer': 'Washer Dryer',
    'kettle':       'Kettle',
    'toaster':      'Toaster',
    'coffee maker': 'Coffee Maker',
    'hair dryer':   'Hair Dryer',
}

AUG_PROBS = {
    'dish washer': 0.3, 'fridge': 0.6,
    'microwave': 0.3,   'washer dryer': 0.3,
    'kettle': 0.3,      'toaster': 0.3,
}

# For UK-DALE appliances not in REDD, use the closest REDD appliance for
# training (zero-shot transfer). Evaluation is still on the actual appliance.
PROXY_APPLIANCE = {
    'kettle':  'microwave',   # both: short-burst, high-power
    'toaster': 'microwave',   # similar power profile
}

# ── Model registry ────────────────────────────────────────────────────────────

# Each entry: (label, color, group)
MODEL_REGISTRY = {
    # Baseline models
    'gru':              ('GRU',              '#4C72B0', 'baseline'),
    'lstm':             ('LSTM',             '#DD8452', 'baseline'),
    'resnet':           ('ResNet',           '#55A868', 'baseline'),
    'tcn':              ('TCN',              '#C44E52', 'baseline'),
    'transformer':      ('Transformer',      '#8172B2', 'baseline'),
    # LNN models
    'standard_lnn':     ('Standard LNN',     '#1a9641', 'lnn'),
    'advanced_lnn':     ('Advanced LNN',     '#fdae61', 'lnn'),
    'attention_lnn':    ('Attention LNN',    '#d73027', 'lnn'),
    'cnn_encoder':      ('CNN+LNN',          '#abd9e9', 'lnn'),
    'transformer_lnn':  ('Transformer+LNN',  '#74add1', 'lnn'),
    'bidirectional_lnn':('Bidir+LNN',        '#4575b4', 'lnn'),
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)[0]


# ── Sequences (BASELINE style) ─────────────────────────────────────────────
# stride=5, target=midpoint, normalise X only

def make_sequences_baseline(df, appliance):
    mains   = df['main'].values
    targets = df[appliance].values
    X, y = [], []
    for i in range(0, len(mains) - WIN, 5):       # stride=5
        X.append(mains[i:i + WIN])
        y.append(targets[i + WIN // 2])            # midpoint
    return (np.array(X, dtype=np.float32).reshape(-1, WIN, 1),
            np.array(y, dtype=np.float32).reshape(-1, 1))


# ── Sequences (LNN style) ──────────────────────────────────────────────────
# stride=1, target=next point after window, normalise both X and y

def make_sequences_lnn(df, appliance):
    mains   = df['main'].values
    targets = df[appliance].values
    X, y = [], []
    for i in range(len(mains) - WIN):              # stride=1
        X.append(mains[i:i + WIN])
        y.append(targets[i + WIN])                 # next point
    return (np.array(X, dtype=np.float32).reshape(-1, WIN, 1),
            np.array(y, dtype=np.float32))


# ── Augmentation (LNN only) ───────────────────────────────────────────────────

def _vscale(x):
    mu, s = 1, 0.2
    tn = truncnorm((mu-2*s-mu)/s, (mu+2*s-mu)/s, loc=mu, scale=s)
    return x * tn.rvs(1)[0]

def _hscale(sig, L):
    mu, s = 1, 0.2
    tn = truncnorm((mu-2*s-mu)/s, (mu+2*s-mu)/s, loc=mu, scale=s)
    sc = tn.rvs(1)[0]
    f  = interp1d(np.arange(len(sig)), sig, kind='linear', fill_value='extrapolate')
    r  = f(np.arange(0, len(sig)-1, sc))
    return r[:L] if len(r) >= L else np.pad(r, (0, L-len(r)))

def augment(X_np, mode):
    out = X_np.copy()
    for i in range(len(out)):
        sig = out[i, :, 0]
        c = np.random.choice(['none','vertical','horizontal','both'], p=[.25,.25,.25,.25]) \
            if mode == 'mixed' else mode
        if c == 'vertical':   sig = _vscale(sig)
        elif c == 'horizontal': sig = _hscale(sig, WIN)
        elif c == 'both':     sig = _vscale(_hscale(sig, WIN))
        out[i, :, 0] = sig
    return out


# ── Dataset ───────────────────────────────────────────────────────────────────

class DS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred, thr):
    yt, yp = y_true.flatten(), y_pred.flatten()
    mae = float(np.mean(np.abs(yt - yp)))
    N, n = 100, len(yt)//100
    sae = float(sum(abs(np.sum(yt[i*N:(i+1)*N])-np.sum(yp[i*N:(i+1)*N]))
                    for i in range(n)) / (N*n)) if n else 0.
    tb = (yt > thr).astype(int); pb = (yp > thr).astype(int)
    tp = int(np.sum((tb==1)&(pb==1))); fp = int(np.sum((tb==0)&(pb==1)))
    fn = int(np.sum((tb==1)&(pb==0)))
    pr = tp/(tp+fp) if (tp+fp) else 0.
    rc = tp/(tp+fn) if (tp+fn) else 0.
    f1 = 2*pr*rc/(pr+rc) if (pr+rc) else 0.
    return {'mae': mae, 'sae': sae, 'f1': float(f1)}


# ── Model builders ────────────────────────────────────────────────────────────

def build_baseline(key):
    if key == 'gru':         return GRUModel(1, 128, 2, 1, bidirectional=True)
    if key == 'lstm':        return LSTMModel(1, 128, 2, 1, bidirectional=True)
    if key == 'resnet':      return ResNetModel(1, 1, [2,2,2], base_width=32)
    if key == 'tcn':         return TCNModel(1, 1, [32,64,128], kernel_size=3, dropout=0.2)
    if key == 'transformer': return SimpleTransformerModel(1, 128, 1, num_layers=3, num_heads=4, dropout=0.1)

def build_lnn(key):
    H = 256
    if key == 'standard_lnn':      return LiquidNetworkModel(1, H, 1)
    if key == 'advanced_lnn':       return AdvancedLiquidNetworkModel(1, H, 1, num_layers=2)
    if key == 'attention_lnn':      return AttentionLiquidNetworkModel(1, H, 1, num_heads=4)
    if key == 'cnn_encoder':        return CNNEncoderLiquidNetworkModel(1, H, 1, num_conv_layers=3)
    if key == 'transformer_lnn':    return TransformerEncoderLiquidNetworkModel(
                                        1, H, 1, num_encoder_layers=1, num_heads=4)
    if key == 'bidirectional_lnn':  return BidirectionalEncoderLiquidNetworkModel(1, H, 1)


# ── Training functions ────────────────────────────────────────────────────────

def _fit(model, tr_loader, va_loader, device, aug_mode, aug_prob, is_lnn):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)
    best_val, best_state, no_imp = float('inf'), None, 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            if is_lnn and aug_mode != 'none' and np.random.rand() < aug_prob:
                xb = torch.FloatTensor(augment(xb.numpy(), aug_mode))
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vl = sum(criterion(model(xb.to(device)), yb.to(device)).item()
                 for xb, yb in va_loader) / len(va_loader)
        scheduler.step(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"    Early stop epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def run_baseline(key, appliance, train_appliance, redd_train, redd_val, ukdale_test, device):
    """train_appliance: column used from REDD for training (may differ for zero-shot)."""
    thr = THRESHOLDS.get(appliance, 10.0)

    X_tr, y_tr = make_sequences_baseline(redd_train,  train_appliance)
    X_va, y_va = make_sequences_baseline(redd_val,    train_appliance)
    X_te, y_te = make_sequences_baseline(ukdale_test, appliance)       # eval on actual

    mu, sig = float(X_tr.mean()), float(X_tr.std())+1e-8
    X_tr = (X_tr-mu)/sig;  X_va = (X_va-mu)/sig;  X_te = (X_te-mu)/sig

    tr = torch.utils.data.DataLoader(DS(X_tr, y_tr), 32, shuffle=True)
    va = torch.utils.data.DataLoader(DS(X_va, y_va), 32, shuffle=False)
    te = torch.utils.data.DataLoader(DS(X_te, y_te), 32, shuffle=False)

    model = build_baseline(key).to(device)
    model = _fit(model, tr, va, device, 'none', 0.0, is_lnn=False)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in te:
            preds.append(model(xb.to(device)).cpu().numpy())
            trues.append(yb.numpy())
    return metrics(np.concatenate(trues), np.concatenate(preds), thr)


def run_lnn(key, appliance, train_appliance, redd_train, redd_val, ukdale_test, device):
    """train_appliance: column used from REDD for training (may differ for zero-shot)."""
    thr = THRESHOLDS.get(appliance, 10.0)

    X_tr, y_tr = make_sequences_lnn(redd_train,  train_appliance)
    X_va, y_va = make_sequences_lnn(redd_val,    train_appliance)
    X_te, y_te = make_sequences_lnn(ukdale_test, appliance)            # eval on actual

    xm, xs = X_tr.mean(), X_tr.std()+1e-8
    ym, ys = y_tr.mean(), y_tr.std()+1e-8
    X_tr=(X_tr-xm)/xs; X_va=(X_va-xm)/xs; X_te=(X_te-xm)/xs
    y_tr=(y_tr-ym)/ys; y_va=(y_va-ym)/ys

    tr = torch.utils.data.DataLoader(
        DS(X_tr, y_tr.reshape(-1,1)), 128, shuffle=True)
    va = torch.utils.data.DataLoader(
        DS(X_va, y_va.reshape(-1,1)), 128, shuffle=False)
    te = torch.utils.data.DataLoader(
        DS(X_te, y_te.reshape(-1,1)), 128, shuffle=False)

    model = build_lnn(key).to(device)
    aug_prob = AUG_PROBS.get(train_appliance, 0.3)
    model = _fit(model, tr, va, device, 'mixed', aug_prob, is_lnn=True)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in te:
            preds.append(model(xb.to(device)).cpu().numpy())
            trues.append(yb.numpy())

    y_pred = np.concatenate(preds) * ys + ym
    y_true = y_te * ys + ym                          # denormalise
    return metrics(y_true, y_pred, thr)


# ── Plots ─────────────────────────────────────────────────────────────────────

def bar_chart(results, metric, ylabel, title, path, apps):
    model_keys = list(MODEL_REGISTRY.keys())
    x, width = np.arange(len(apps)), 0.07
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, mk in enumerate(model_keys):
        label, color, _ = MODEL_REGISTRY[mk]
        vals   = [results.get(mk, {}).get(a, {}).get(metric, np.nan) for a in apps]
        offset = (i - len(model_keys)/2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()*1.01,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([APP_LABELS.get(a, a) for a in apps])
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved → {path}")


def print_table(results, apps):
    for metric, label in [('f1','F1'), ('mae','MAE'), ('sae','SAE')]:
        print(f"\n{'='*75}")
        print(f"  {label} — Train: REDD  /  Test: UK-DALE")
        print(f"{'='*75}")
        col_w = 14
        header = f"  {'Model':<22}" + ''.join(f"{APP_LABELS.get(a,a):<{col_w}}" for a in apps) + "Avg"
        print(header); print('─'*75)
        for mk, (lbl, _, _) in MODEL_REGISTRY.items():
            vals = [results.get(mk,{}).get(a,{}).get(metric, np.nan) for a in apps]
            avg  = np.nanmean(vals)
            print(f"  {lbl:<22}" + ''.join(f"{v:<{col_w}.4f}" for v in vals) + f"{avg:.4f}")


# ── Shared output directory ───────────────────────────────────────────────────

SAVE_DIR = os.path.join('results', 'ukdale_eval')


def _get_apps(redd_train, ukdale_test):
    ukdale_cols = set(ukdale_test.columns) - {'main'}
    redd_cols   = set(redd_train.columns)  - {'main'}
    direct      = sorted(ukdale_cols & redd_cols)
    zero_shot   = sorted(ukdale_cols - redd_cols)

    print(f"REDD appliances            : {sorted(redd_cols)}")
    print(f"UK-DALE appliances         : {sorted(ukdale_cols)}")
    print(f"Direct (REDD→UK-DALE)      : {direct}")
    if zero_shot:
        for a in zero_shot:
            proxy = PROXY_APPLIANCE.get(a)
            if proxy:
                print(f"Zero-shot (proxy)          : {a}  ← trained on '{proxy}'")
            else:
                print(f"Skipped (no proxy defined) : {a}")

    # All UK-DALE appliances that have either a direct match or a proxy
    apps = sorted(a for a in ukdale_cols
                  if a in redd_cols or a in PROXY_APPLIANCE)
    return apps


def run_one_model(mk, redd_train, redd_val, ukdale_test, apps, device):
    """Train one model on all appliances and save its JSON."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    label, _, group = MODEL_REGISTRY[mk]

    print(f"\n{'#'*65}")
    print(f"#  {label}  [{group}]  (train: REDD → test: UK-DALE)")
    print(f"{'#'*65}")

    redd_cols = set(redd_train.columns) - {'main'}
    results = {}
    for app in apps:
        train_app = app if app in redd_cols else PROXY_APPLIANCE.get(app, app)
        tag = f"zero-shot via '{train_app}'" if train_app != app else "direct"
        print(f"\n  ▶  {app}  [{tag}]")
        try:
            if group == 'baseline':
                m = run_baseline(mk, app, train_app, redd_train, redd_val, ukdale_test, device)
            else:
                m = run_lnn(mk, app, train_app, redd_train, redd_val, ukdale_test, device)
            results[app] = m
            print(f"  ✅ {label:<22} | {app:<15} | "
                  f"F1={m['f1']:.4f}  MAE={m['mae']:.2f}  SAE={m['sae']:.4f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ❌ {label} / {app}: {e}")

    json_path = os.path.join(SAVE_DIR, f'{mk}.json')
    with open(json_path, 'w') as f:
        json.dump({'model': mk, 'label': label, 'group': group,
                   'epochs': EPOCHS, 'appliances': apps, 'results': results}, f, indent=2)
    print(f"\n  JSON saved → {json_path}")


def load_all_results():
    all_results = {}
    apps = []
    for mk in MODEL_REGISTRY:
        path = os.path.join(SAVE_DIR, f'{mk}.json')
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            all_results[mk] = data['results']
            if not apps:
                apps = data.get('appliances', [])
            print(f"  Loaded {MODEL_REGISTRY[mk][0]} from {path}")
        else:
            print(f"  Missing: {path}  (run --model {mk} first)")
    return all_results, apps


def generate_plots(all_results, apps):
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("\nGenerating graphs...")
    for metric, ylabel in [('mae','MAE (W)'), ('sae','SAE'), ('f1','F1 Score')]:
        bar_chart(all_results, metric, ylabel,
                  f'{ylabel} — Trained REDD / Tested UK-DALE',
                  os.path.join(SAVE_DIR, f'{metric}_comparison.png'), apps)
    print_table(all_results, apps)

    json_path = os.path.join(SAVE_DIR, 'results_summary.json')
    with open(json_path, 'w') as f:
        json.dump({'train': 'REDD', 'test': 'UK-DALE',
                   'appliances': apps, 'results': all_results}, f, indent=2)
    print(f"\nCombined JSON → {json_path}")
    print(f"All graphs in  → {SAVE_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_model_keys = list(MODEL_REGISTRY.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        choices=all_model_keys + ['all'],
        default='all',
        help='Which model to train and evaluate.'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Skip training — load saved JSONs and regenerate graphs.'
    )
    args = parser.parse_args()

    # ── Plot-only mode ──
    if args.plot:
        print("Plot mode — loading saved results...")
        all_results, apps = load_all_results()
        if not any(all_results.values()):
            print("No results found. Run at least one model first.")
            sys.exit(1)
        generate_plots(all_results, apps)
        return

    # ── Check files ──
    for path in (REDD_TRAIN_PKL, REDD_VAL_PKL, UKDALE_TEST_PKL):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found.")
            if 'ukdale' in path:
                print("  → Run preprocess_ukdale_to_pkl.py first.")
            sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading data...")
    redd_train  = load_pkl(REDD_TRAIN_PKL)
    redd_val    = load_pkl(REDD_VAL_PKL)
    ukdale_test = load_pkl(UKDALE_TEST_PKL)

    apps = _get_apps(redd_train, ukdale_test)
    if not apps:
        print("ERROR: No common appliances found.")
        sys.exit(1)

    models_to_run = all_model_keys if args.model == 'all' else [args.model]

    for mk in models_to_run:
        run_one_model(mk, redd_train, redd_val, ukdale_test, apps, device)

    # Auto-generate plots if all models done
    all_done = all(
        os.path.exists(os.path.join(SAVE_DIR, f'{mk}.json'))
        for mk in all_model_keys
    )
    if all_done:
        print("\nAll models complete — generating combined graphs...")
        all_results, apps = load_all_results()
        generate_plots(all_results, apps)
    else:
        missing = [mk for mk in all_model_keys
                   if not os.path.exists(os.path.join(SAVE_DIR, f'{mk}.json'))]
        print(f"\nStill missing: {missing}")
        print("Run those models then:  !python evaluate_on_ukdale.py --plot")


if __name__ == '__main__':
    main()
