"""
Visualization of Causal Liquid Neural Network Mathematics
Generates figures explaining the key mathematical concepts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def visualize_causal_cell_architecture():
    """Visualize the Causal Liquid Cell architecture"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Causal Liquid Cell Architecture',
            fontsize=18, fontweight='bold', ha='center')

    # Input
    input_box = FancyBboxPatch((0.5, 7), 1, 0.6, boxstyle="round,pad=0.1",
                               edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1, 7.3, r'$x_t$', fontsize=14, ha='center', fontweight='bold')

    # Event Detector
    event_box = FancyBboxPatch((2.5, 7.5), 1.5, 0.8, boxstyle="round,pad=0.1",
                               edgecolor='red', facecolor='#ffcccc', linewidth=2)
    ax.add_patch(event_box)
    ax.text(3.25, 8.1, 'Event Detector', fontsize=10, ha='center', fontweight='bold')
    ax.text(3.25, 7.7, r'$e_t = \sigma(W_e \cdot x_t)$', fontsize=9, ha='center')

    # Input Processing
    input_proc_box = FancyBboxPatch((2.5, 6), 1.5, 0.8, boxstyle="round,pad=0.1",
                                    edgecolor='green', facecolor='#ccffcc', linewidth=2)
    ax.add_patch(input_proc_box)
    ax.text(3.25, 6.6, 'Input Process', fontsize=10, ha='center', fontweight='bold')
    ax.text(3.25, 6.2, r'$i_t = \tanh(W_{in} \cdot x_t)$', fontsize=9, ha='center')

    # Previous Hidden State
    prev_hidden = FancyBboxPatch((0.5, 4), 1, 0.6, boxstyle="round,pad=0.1",
                                edgecolor='purple', facecolor='#e6ccff', linewidth=2)
    ax.add_patch(prev_hidden)
    ax.text(1, 4.3, r'$h_{t-1}$', fontsize=14, ha='center', fontweight='bold')

    # Causal Attention
    attention_box = FancyBboxPatch((2.5, 4), 1.5, 0.8, boxstyle="round,pad=0.1",
                                   edgecolor='orange', facecolor='#ffedcc', linewidth=2)
    ax.add_patch(attention_box)
    ax.text(3.25, 4.6, 'Causal Attention', fontsize=10, ha='center', fontweight='bold')
    ax.text(3.25, 4.2, r'$\alpha_t = \sigma(W_a \cdot h_{t-1})$', fontsize=9, ha='center')

    # Recurrent Processing
    recurrent_box = FancyBboxPatch((5, 4), 2, 0.8, boxstyle="round,pad=0.1",
                                   edgecolor='brown', facecolor='#ffe6cc', linewidth=2)
    ax.add_patch(recurrent_box)
    ax.text(6, 4.6, 'Recurrent Process', fontsize=10, ha='center', fontweight='bold')
    ax.text(6, 4.2, r'$r_t = \tanh(W_{rec} \cdot (\alpha_t \odot h_{t-1}))$', fontsize=9, ha='center')

    # Liquid Dynamics
    dynamics_box = FancyBboxPatch((5, 2), 2.5, 1, boxstyle="round,pad=0.1",
                                  edgecolor='darkblue', facecolor='#cce6ff', linewidth=3)
    ax.add_patch(dynamics_box)
    ax.text(6.25, 2.7, 'Liquid Dynamics', fontsize=11, ha='center', fontweight='bold')
    ax.text(6.25, 2.4, r'$\frac{dh_t}{dt} = \frac{-h_{t-1} + i_t + r_t}{\tau}$', fontsize=10, ha='center')

    # Event Weighting
    event_weight_box = FancyBboxPatch((5, 0.5), 2.5, 0.8, boxstyle="round,pad=0.1",
                                      edgecolor='red', facecolor='#ffcccc', linewidth=2)
    ax.add_patch(event_weight_box)
    ax.text(6.25, 1.1, 'Event Weighting', fontsize=10, ha='center', fontweight='bold')
    ax.text(6.25, 0.7, r'$(1 + e_t)$', fontsize=10, ha='center')

    # Final Update
    final_box = FancyBboxPatch((8.5, 2), 1.2, 1, boxstyle="round,pad=0.1",
                               edgecolor='darkgreen', facecolor='#ccffcc', linewidth=3)
    ax.add_patch(final_box)
    ax.text(9.1, 2.7, 'Final State', fontsize=11, ha='center', fontweight='bold')
    ax.text(9.1, 2.3, r'$h_t$', fontsize=14, ha='center', fontweight='bold')

    # Arrows
    arrows = [
        ((1.5, 7.3), (2.5, 7.9)),  # x_t to event
        ((1.5, 7.3), (2.5, 6.4)),  # x_t to input process
        ((1.5, 4.3), (2.5, 4.4)),  # h_t-1 to attention
        ((4, 4.4), (5, 4.4)),      # attention to recurrent
        ((4, 6.4), (5, 2.7)),      # input to dynamics
        ((7, 4.4), (5.5, 2.7)),    # recurrent to dynamics
        ((4, 7.9), (6.25, 1.3)),   # event to weighting
        ((6.25, 2), (6.25, 1.3)),  # dynamics to weighting
        ((6.25, 0.5), (8.5, 2.5)), # weighting to final
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='black', alpha=0.6)
        ax.add_patch(arrow)

    # Add equation at bottom
    eq_text = r'$h_t = h_{t-1} + \Delta t \cdot \frac{dh_t}{dt} \cdot (1 + e_t)$'
    ax.text(5, 0.1, eq_text, fontsize=14, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig('causal_cell_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: causal_cell_architecture.png")
    plt.close()


def visualize_event_weighting():
    """Visualize event detection and weighting mechanism"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Generate sample data
    time = np.linspace(0, 100, 1000)

    # Appliance power signal (with state changes)
    power = np.zeros_like(time)
    power[200:400] = 100  # ON period
    power[600:800] = 100  # ON period

    # Add noise
    power += np.random.normal(0, 5, len(time))

    # Detect events (state changes)
    events = np.abs(np.diff(power, prepend=power[0])) > 20
    event_weights = events.astype(float)

    # Event-weighted importance
    importance = 1 + 2 * event_weights  # lambda = 2

    # Plot 1: Appliance Power Signal
    axes[0, 0].plot(time, power, 'b-', linewidth=1.5, label='Appliance Power')
    axes[0, 0].scatter(time[events], power[events], c='red', s=100,
                       marker='*', label='Detected Events', zorder=5)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Power (W)')
    axes[0, 0].set_title('(a) Appliance Power with State Changes', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Event Weights
    axes[0, 1].stem(time, event_weights, linefmt='r-', markerfmt='ro',
                    basefmt='k-', label='Event Weight $e_t$')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Event Weight')
    axes[0, 1].set_title('(b) Event Detection Signal', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-0.1, 1.5)

    # Plot 3: Sample Importance
    axes[1, 0].plot(time, importance, 'g-', linewidth=2, label=r'$w = 1 + \lambda \cdot e_t$')
    axes[1, 0].axhline(1, color='gray', linestyle='--', label='Base Weight')
    axes[1, 0].axhline(3, color='red', linestyle='--', alpha=0.5, label='Event Weight (λ=2)')
    axes[1, 0].fill_between(time, 1, importance, where=(importance > 1),
                            alpha=0.3, color='red', label='Extra Importance')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Sample Weight')
    axes[1, 0].set_title(r'(c) Sample Importance: $w_i = 1 + \lambda \cdot e_i$', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Loss Weighting Comparison
    prediction_errors = np.random.uniform(0, 50, len(time))
    standard_loss = prediction_errors ** 2
    causal_loss = importance * (prediction_errors ** 2)

    axes[1, 1].plot(time, standard_loss, 'b-', alpha=0.5, linewidth=1.5, label='Standard Loss')
    axes[1, 1].plot(time, causal_loss, 'r-', linewidth=2, label='Causal Weighted Loss')
    axes[1, 1].fill_between(time, standard_loss, causal_loss,
                            where=(causal_loss > standard_loss),
                            alpha=0.3, color='red', label='Extra Loss Weight')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Loss Value')
    axes[1, 1].set_title(r'(d) Loss Comparison: $L_{causal} = w_i \cdot (y - \hat{y})^2$', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Event Detection and Weighting Mechanism', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('event_weighting_mechanism.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: event_weighting_mechanism.png")
    plt.close()


def visualize_liquid_dynamics():
    """Visualize liquid dynamics differential equation"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time constants
    taus = [0.5, 1.0, 2.0, 5.0]
    colors = ['red', 'orange', 'green', 'blue']
    time = np.linspace(0, 10, 1000)

    # Plot 1: Effect of time constant τ
    for tau, color in zip(taus, colors):
        # Solve dh/dt = (-h + 1) / tau
        h = 1 - np.exp(-time / tau)
        axes[0, 0].plot(time, h, color=color, linewidth=2, label=f'τ = {tau}')

    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Hidden State h(t)')
    axes[0, 0].set_title(r'(a) Effect of Time Constant $\tau$ on Dynamics', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(1, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Euler integration with different dt
    dts = [0.05, 0.1, 0.5, 1.0]
    tau = 1.0

    for dt, color in zip(dts, colors):
        t_discrete = np.arange(0, 10, dt)
        h = np.zeros_like(t_discrete)
        for i in range(1, len(t_discrete)):
            dhdt = (-h[i-1] + 1) / tau
            h[i] = h[i-1] + dt * dhdt
        axes[0, 1].plot(t_discrete, h, 'o-', color=color, linewidth=1.5,
                       markersize=4, label=f'Δt = {dt}')

    # True solution
    h_true = 1 - np.exp(-time / tau)
    axes[0, 1].plot(time, h_true, 'k--', linewidth=2, label='True Solution', alpha=0.5)

    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Hidden State h(t)')
    axes[0, 1].set_title(r'(b) Euler Integration with Different $\Delta t$', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Event amplification
    tau = 1.0
    dt = 0.1
    t_discrete = np.arange(0, 20, dt)
    h_standard = np.zeros_like(t_discrete)
    h_causal = np.zeros_like(t_discrete)

    # Events at t=5 and t=15
    event_weights = np.zeros_like(t_discrete)
    event_weights[int(5/dt):int(7/dt)] = 1.0
    event_weights[int(15/dt):int(17/dt)] = 1.0

    for i in range(1, len(t_discrete)):
        dhdt = (-h_standard[i-1] + 1) / tau
        h_standard[i] = h_standard[i-1] + dt * dhdt

        # With event weighting
        h_causal[i] = h_causal[i-1] + dt * dhdt * (1 + event_weights[i])

    axes[1, 0].plot(t_discrete, h_standard, 'b-', linewidth=2, label='Standard Update')
    axes[1, 0].plot(t_discrete, h_causal, 'r-', linewidth=2, label='Event-Weighted Update')
    axes[1, 0].fill_between(t_discrete, 0, event_weights * max(h_causal),
                            alpha=0.2, color='yellow', label='Event Periods')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Hidden State h(t)')
    axes[1, 0].set_title(r'(c) Event Amplification: $h_t = h_{t-1} + \Delta t \cdot \frac{dh}{dt} \cdot (1+e_t)$',
                        fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Phase portrait
    h_vals = np.linspace(-1, 2, 20)
    tau_vals = [0.5, 1.0, 2.0]

    for tau, color in zip(tau_vals[:3], colors[:3]):
        dhdt = (-h_vals + 1) / tau
        axes[1, 1].plot(h_vals, dhdt, color=color, linewidth=2, label=f'τ = {tau}')

    axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].axvline(1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Equilibrium h=1')
    axes[1, 1].set_xlabel('Hidden State h')
    axes[1, 1].set_ylabel(r'$\frac{dh}{dt}$')
    axes[1, 1].set_title(r'(d) Phase Portrait: $\frac{dh}{dt} = \frac{-h + 1}{\tau}$', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Liquid Dynamics and Time Evolution', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('liquid_dynamics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: liquid_dynamics.png")
    plt.close()


def visualize_granger_causality():
    """Visualize Granger causality concept"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Generate synthetic data with causal relationship
    np.random.seed(42)
    n = 200

    # X causes Y with lag
    X = np.cumsum(np.random.randn(n)) * 0.5
    Y = np.zeros(n)
    for t in range(5, n):
        Y[t] = 0.6 * Y[t-1] + 0.4 * X[t-3] + np.random.randn() * 0.3

    time = np.arange(n)

    # Plot 1: Time series
    axes[0, 0].plot(time, X, 'b-', linewidth=1.5, label='X (Aggregate Power)', alpha=0.7)
    axes[0, 0].plot(time, Y, 'r-', linewidth=1.5, label='Y (Appliance Power)', alpha=0.7)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Power')
    axes[0, 0].set_title('(a) Time Series: X and Y', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Lagged scatter
    lag = 3
    axes[0, 1].scatter(X[:-lag], Y[lag:], alpha=0.5, s=30)
    axes[0, 1].set_xlabel(f'X(t-{lag})')
    axes[0, 1].set_ylabel('Y(t)')
    axes[0, 1].set_title(f'(b) Lagged Relationship: X(t-{lag}) vs Y(t)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Fit line
    z = np.polyfit(X[:-lag], Y[lag:], 1)
    p = np.poly1d(z)
    x_line = np.linspace(X.min(), X.max(), 100)
    axes[0, 1].plot(x_line, p(x_line), 'r--', linewidth=2,
                   label=f'Fit: y = {z[0]:.2f}x + {z[1]:.2f}')
    axes[0, 1].legend()

    # Plot 3: Granger causality concept
    axes[1, 0].text(0.5, 0.85, 'Granger Causality Test', fontsize=16,
                   ha='center', fontweight='bold', transform=axes[1, 0].transAxes)

    # Restricted model
    axes[1, 0].text(0.1, 0.65, 'Restricted Model (Y only):', fontsize=12,
                   fontweight='bold', transform=axes[1, 0].transAxes)
    axes[1, 0].text(0.1, 0.55, r'$y_t = \sum_{k=1}^p \alpha_k \cdot y_{t-k} + \epsilon_t^{(r)}$',
                   fontsize=11, transform=axes[1, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Full model
    axes[1, 0].text(0.1, 0.35, 'Full Model (Y and X):', fontsize=12,
                   fontweight='bold', transform=axes[1, 0].transAxes)
    axes[1, 0].text(0.1, 0.20, r'$y_t = \sum_{k=1}^p \beta_k \cdot y_{t-k} + \sum_{k=1}^p \gamma_k \cdot x_{t-k} + \epsilon_t^{(f)}$',
                   fontsize=10, transform=axes[1, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Score
    axes[1, 0].text(0.5, 0.05, r'$GC = \frac{RSS_{restricted} - RSS_{full}}{RSS_{full}}$',
                   fontsize=14, ha='center', transform=axes[1, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    axes[1, 0].axis('off')

    # Plot 4: RSS comparison (bar chart)
    # Simulate RSS values
    from sklearn.linear_model import LinearRegression

    # Prepare data
    max_lag = 5
    X_lagged = np.column_stack([Y[i:-(max_lag-i) if max_lag-i > 0 else None]
                                for i in range(max_lag)])
    X_with_cause = np.column_stack([X_lagged,
                                    np.column_stack([X[i:-(max_lag-i) if max_lag-i > 0 else None]
                                                    for i in range(max_lag)])])
    y_target = Y[max_lag:]

    # Restricted model
    model_restricted = LinearRegression().fit(X_lagged, y_target)
    rss_restricted = np.sum((y_target - model_restricted.predict(X_lagged)) ** 2)

    # Full model
    model_full = LinearRegression().fit(X_with_cause, y_target)
    rss_full = np.sum((y_target - model_full.predict(X_with_cause)) ** 2)

    gc_score = (rss_restricted - rss_full) / rss_full

    bars = axes[1, 1].bar(['Restricted\n(Y only)', 'Full\n(Y + X)'],
                         [rss_restricted, rss_full],
                         color=['lightblue', 'lightgreen'], edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Residual Sum of Squares (RSS)')
    axes[1, 1].set_title(f'(d) RSS Comparison\nGC Score = {gc_score:.3f}', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontweight='bold')

    # Add arrow showing improvement
    axes[1, 1].annotate('', xy=(1, rss_full), xytext=(1, rss_restricted),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    axes[1, 1].text(1.1, (rss_restricted + rss_full)/2, 'Improvement\nfrom X',
                   fontsize=10, color='red', fontweight='bold')

    plt.suptitle('Granger Causality: Does X (Aggregate) Cause Y (Appliance)?',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('granger_causality.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: granger_causality.png")
    plt.close()


def visualize_f1_improvement():
    """Visualize how causal learning improves F1 score"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Generate synthetic predictions
    np.random.seed(42)
    n = 1000

    # Ground truth (appliance on/off)
    true_signal = np.zeros(n)
    true_signal[200:400] = 100
    true_signal[600:800] = 100
    true_signal += np.random.normal(0, 5, n)

    # Standard model predictions (poor at transitions)
    std_predictions = true_signal.copy()
    std_predictions[:220] *= 0.3  # Missed early part
    std_predictions[380:420] += 40  # False positives
    std_predictions[600:650] *= 0.4  # Missed early part
    std_predictions += np.random.normal(0, 15, n)

    # Causal model predictions (better at transitions)
    causal_predictions = true_signal.copy()
    causal_predictions += np.random.normal(0, 10, n)

    time = np.arange(n)
    threshold = 30

    # Plot 1: Signal comparison
    axes[0, 0].plot(time, true_signal, 'g-', linewidth=2, label='True Signal', alpha=0.7)
    axes[0, 0].plot(time, std_predictions, 'b--', linewidth=1.5, label='Standard LNN', alpha=0.7)
    axes[0, 0].plot(time, causal_predictions, 'r-', linewidth=1.5, label='Causal LNN', alpha=0.7)
    axes[0, 0].axhline(threshold, color='black', linestyle=':', label='Threshold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Power (W)')
    axes[0, 0].set_title('(a) Predictions Comparison', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Compute confusion matrix elements
    def compute_metrics(predictions, truth, threshold):
        pred_binary = predictions > threshold
        true_binary = truth > threshold

        TP = np.sum(pred_binary & true_binary)
        FP = np.sum(pred_binary & ~true_binary)
        FN = np.sum(~pred_binary & true_binary)
        TN = np.sum(~pred_binary & ~true_binary)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
                'precision': precision, 'recall': recall, 'f1': f1}

    std_metrics = compute_metrics(std_predictions, true_signal, threshold)
    causal_metrics = compute_metrics(causal_predictions, true_signal, threshold)

    # Plot 2: Confusion matrices
    conf_std = np.array([[std_metrics['TN'], std_metrics['FP']],
                         [std_metrics['FN'], std_metrics['TP']]])
    conf_causal = np.array([[causal_metrics['TN'], causal_metrics['FP']],
                            [causal_metrics['FN'], causal_metrics['TP']]])

    # Standard LNN confusion matrix
    im1 = axes[0, 1].imshow(conf_std, cmap='Blues', aspect='auto')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_yticks([0, 1])
    axes[0, 1].set_xticklabels(['Pred OFF', 'Pred ON'])
    axes[0, 1].set_yticklabels(['True OFF', 'True ON'])
    axes[0, 1].set_title(f'(b) Standard LNN\nF1 = {std_metrics["f1"]:.3f}', fontweight='bold')

    for i in range(2):
        for j in range(2):
            axes[0, 1].text(j, i, f'{conf_std[i, j]}',
                           ha='center', va='center', fontsize=14, fontweight='bold')

    # Plot 3: Causal LNN confusion matrix
    im2 = axes[1, 0].imshow(conf_causal, cmap='Greens', aspect='auto')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Pred OFF', 'Pred ON'])
    axes[1, 0].set_yticklabels(['True OFF', 'True ON'])
    axes[1, 0].set_title(f'(c) Causal LNN\nF1 = {causal_metrics["f1"]:.3f}', fontweight='bold')

    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, f'{conf_causal[i, j]}',
                           ha='center', va='center', fontsize=14, fontweight='bold')

    # Plot 4: Metrics comparison
    metrics_names = ['Precision', 'Recall', 'F1']
    std_values = [std_metrics['precision'], std_metrics['recall'], std_metrics['f1']]
    causal_values = [causal_metrics['precision'], causal_metrics['recall'], causal_metrics['f1']]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = axes[1, 1].bar(x - width/2, std_values, width, label='Standard LNN',
                          color='lightblue', edgecolor='black', linewidth=2)
    bars2 = axes[1, 1].bar(x + width/2, causal_values, width, label='Causal LNN',
                          color='lightgreen', edgecolor='black', linewidth=2)

    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('(d) Metrics Comparison', fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim(0, 1.0)

    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('F1 Score Improvement: Standard vs Causal LNN',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('f1_improvement.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: f1_improvement.png")
    plt.close()


def visualize_temporal_causality():
    """Visualize temporal causality constraint"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Causal mask matrix
    T = 10
    causal_mask = np.tril(np.ones((T, T)))

    axes[0].imshow(causal_mask, cmap='RdYlGn', aspect='auto')
    axes[0].set_xlabel('Time Step (attending to)')
    axes[0].set_ylabel('Time Step (current)')
    axes[0].set_title('(a) Causal Attention Mask\n(Green = Allowed, Red = Blocked)',
                     fontweight='bold')

    # Add grid
    for i in range(T+1):
        axes[0].axhline(i-0.5, color='black', linewidth=0.5)
        axes[0].axvline(i-0.5, color='black', linewidth=0.5)

    # Add labels
    for i in range(T):
        for j in range(T):
            if causal_mask[i, j] == 1:
                axes[0].text(j, i, '✓', ha='center', va='center',
                           fontsize=12, fontweight='bold', color='darkgreen')
            else:
                axes[0].text(j, i, '✗', ha='center', va='center',
                           fontsize=12, fontweight='bold', color='darkred')

    # Information flow diagram
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10)
    axes[1].axis('off')

    # Title
    axes[1].text(5, 9.5, '(b) Temporal Causality Constraint',
                fontsize=14, fontweight='bold', ha='center')

    # Past
    past_box = FancyBboxPatch((0.5, 6), 2, 1.5, boxstyle="round,pad=0.1",
                              edgecolor='green', facecolor='lightgreen', linewidth=3)
    axes[1].add_patch(past_box)
    axes[1].text(1.5, 7.2, 'PAST', fontsize=12, ha='center', fontweight='bold')
    axes[1].text(1.5, 6.5, r'$x_{t-2}, x_{t-1}$', fontsize=11, ha='center')
    axes[1].text(1.5, 6.2, r'$h_{t-1}$', fontsize=11, ha='center')

    # Present
    present_box = FancyBboxPatch((4, 6), 2, 1.5, boxstyle="round,pad=0.1",
                                 edgecolor='blue', facecolor='lightblue', linewidth=3)
    axes[1].add_patch(present_box)
    axes[1].text(5, 7.2, 'PRESENT', fontsize=12, ha='center', fontweight='bold')
    axes[1].text(5, 6.5, r'$x_t$', fontsize=11, ha='center')
    axes[1].text(5, 6.2, r'$h_t$', fontsize=11, ha='center')

    # Future
    future_box = FancyBboxPatch((7.5, 6), 2, 1.5, boxstyle="round,pad=0.1",
                                edgecolor='red', facecolor='#ffcccc', linewidth=3)
    axes[1].add_patch(future_box)
    axes[1].text(8.5, 7.2, 'FUTURE', fontsize=12, ha='center', fontweight='bold')
    axes[1].text(8.5, 6.5, r'$x_{t+1}, x_{t+2}$', fontsize=11, ha='center')
    axes[1].text(8.5, 6.2, 'BLOCKED', fontsize=10, ha='center',
                fontweight='bold', color='red')

    # Arrows (allowed)
    arrow1 = FancyArrowPatch((2.5, 6.75), (4, 6.75), arrowstyle='->',
                            mutation_scale=30, linewidth=3, color='green')
    axes[1].add_patch(arrow1)
    axes[1].text(3.25, 7.4, 'CAUSAL', fontsize=10, ha='center',
                fontweight='bold', color='green')

    # Blocked arrow
    axes[1].plot([6, 7.5], [6.75, 6.75], 'r-', linewidth=3)
    axes[1].plot([6.5, 7], [6.4, 7.1], 'r-', linewidth=3)
    axes[1].plot([6.5, 7], [7.1, 6.4], 'r-', linewidth=3)
    axes[1].text(6.75, 7.4, 'BLOCKED', fontsize=10, ha='center',
                fontweight='bold', color='red')

    # Equation
    eq = r'$h_t = f(x_t, h_{t-1})$  ← Causal' + '\n' + r'$h_t \neq f(x_{t+1})$  ← Blocked'
    axes[1].text(5, 4.5, eq, fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Key principle
    axes[1].text(5, 2.5, 'Cause precedes Effect', fontsize=14, ha='center',
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1].text(5, 1.5, 'Past and Present influence Current state',
                fontsize=11, ha='center')
    axes[1].text(5, 0.8, 'Future cannot influence Current state',
                fontsize=11, ha='center', style='italic')

    plt.suptitle('Temporal Causality in Causal LNN', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('temporal_causality.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: temporal_causality.png")
    plt.close()


if __name__ == "__main__":
    print("Generating Causal LNN Mathematical Visualizations...")
    print("=" * 70)

    visualize_causal_cell_architecture()
    visualize_event_weighting()
    visualize_liquid_dynamics()
    visualize_granger_causality()
    visualize_f1_improvement()
    visualize_temporal_causality()

    print("=" * 70)
    print("All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  1. causal_cell_architecture.png - Architecture diagram")
    print("  2. event_weighting_mechanism.png - Event detection and weighting")
    print("  3. liquid_dynamics.png - Differential equations and time evolution")
    print("  4. granger_causality.png - Causal relationship testing")
    print("  5. f1_improvement.png - Performance comparison")
    print("  6. temporal_causality.png - Temporal causality constraints")
