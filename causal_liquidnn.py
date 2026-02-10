"""
Causal Liquid Neural Network for NILM
Incorporates causal learning principles into LNN architecture
"""

import torch
import torch.nn as nn
import numpy as np


class CausalLiquidCell(nn.Module):
    """
    Causal Liquid Neural Network Cell with temporal causality constraints

    Key causal principles:
    1. Temporal causality: Only past information influences present
    2. Causal masking: Attention limited to past time steps
    3. Event-driven dynamics: State changes based on causal events
    """

    def __init__(self, input_size, hidden_size, dt=0.1):
        """
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            dt: Time constant for liquid dynamics
        """
        super(CausalLiquidCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt

        # Input to hidden connections (causal)
        self.W_in = nn.Linear(input_size, hidden_size)

        # Recurrent connections with causal constraints
        self.W_rec = nn.Linear(hidden_size, hidden_size)

        # Time constant parameters (learnable)
        self.tau = nn.Parameter(torch.ones(hidden_size))

        # Causal attention mechanism
        self.causal_attention = nn.Linear(hidden_size, hidden_size)

        # Event detector (for causal event weighting)
        # Takes both current input and previous input to detect changes
        self.event_detector = nn.Linear(input_size * 2, 1)

        # Store previous input for event detection
        self.prev_input = None

    def forward(self, x, hidden):
        """
        Forward pass with causal dynamics

        Args:
            x: Input at current time step (batch_size, input_size)
            hidden: Previous hidden state (batch_size, hidden_size)

        Returns:
            new_hidden: Updated hidden state
            event_weight: Causal event weight for this time step
        """
        # Detect causal events by comparing current and previous inputs
        if self.prev_input is None:
            # First time step: no event detected
            event_weight = torch.zeros(x.size(0), 1, device=x.device)
        else:
            # Detect events from temporal changes
            # Concatenate current and previous inputs
            event_input = torch.cat([x, self.prev_input], dim=1)
            event_weight = torch.sigmoid(self.event_detector(event_input))

        # Update previous input for next time step
        self.prev_input = x.detach()

        # Compute input contribution
        input_contrib = torch.tanh(self.W_in(x))

        # Compute recurrent contribution with causal attention
        attention_weights = torch.sigmoid(self.causal_attention(hidden))
        recurrent_contrib = torch.tanh(self.W_rec(hidden * attention_weights))

        # Liquid dynamics with learned time constants
        # dx/dt = (-x + f(inputs)) / tau
        tau_clamped = torch.clamp(self.tau, min=0.1, max=10.0)
        dhidden_dt = (-hidden + input_contrib + recurrent_contrib) / tau_clamped

        # Euler integration with causal event weighting
        # Event weighting amplifies updates when input changes significantly
        new_hidden = hidden + self.dt * dhidden_dt * (1.0 + event_weight)

        return new_hidden, event_weight


class CausalLiquidNetworkModel(nn.Module):
    """
    Causal Liquid Neural Network for NILM with temporal causality

    Architecture:
    1. Input processing
    2. Causal liquid cell with temporal constraints
    3. Causal event detection
    4. Output prediction
    """

    def __init__(self, input_size=1, hidden_size=128, output_size=1, dt=0.1):
        """
        Args:
            input_size: Input dimension (aggregate power)
            hidden_size: Hidden state dimension
            output_size: Output dimension (appliance power)
            dt: Time constant for liquid dynamics
        """
        super(CausalLiquidNetworkModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Causal liquid cell
        self.liquid_cell = CausalLiquidCell(input_size, hidden_size, dt)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Causal event accumulator (tracks causal influence over time)
        self.event_accumulator = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        """
        Forward pass through causal liquid network

        Args:
            x: Input sequence (batch_size, seq_len, input_size)

        Returns:
            output: Predictions (batch_size, output_size)
            event_weights: Causal event weights (batch_size, seq_len)
        """
        batch_size, seq_len, _ = x.size()

        # Initialize hidden state
        hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Reset previous input for new sequence
        self.liquid_cell.prev_input = None

        # Event accumulator for tracking causal influence
        event_accumulation = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Store event weights for analysis
        event_weights = []

        # Process sequence with causal temporal ordering
        for t in range(seq_len):
            # Current input (cause)
            x_t = x[:, t, :]

            # Update hidden state with causal dynamics
            hidden, event_weight = self.liquid_cell(x_t, hidden)

            # Accumulate causal events
            event_accumulation = 0.9 * event_accumulation + 0.1 * torch.tanh(
                self.event_accumulator(hidden)
            ) * event_weight

            # Store event weights
            event_weights.append(event_weight)

        # Combine liquid state with causal event accumulation
        final_state = hidden + event_accumulation

        # Generate output (effect)
        output = self.output_layer(final_state)

        # Stack event weights
        event_weights = torch.cat(event_weights, dim=1)  # (batch_size, seq_len)

        return output, event_weights


class AdvancedCausalLiquidNetworkModel(nn.Module):
    """
    Advanced Causal Liquid Neural Network with multi-layer causal processing

    Enhancements:
    1. Multiple causal liquid layers
    2. Skip connections for direct causal paths
    3. Hierarchical temporal causality
    """

    def __init__(self, input_size=1, hidden_size=128, output_size=1,
                 num_layers=2, dt=0.1):
        """
        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
            output_size: Output dimension
            num_layers: Number of causal liquid layers
            dt: Time constant for liquid dynamics
        """
        super(AdvancedCausalLiquidNetworkModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Stack of causal liquid cells
        self.liquid_cells = nn.ModuleList([
            CausalLiquidCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                dt
            )
            for i in range(num_layers)
        ])

        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        # Skip connections (direct causal paths)
        self.skip_connections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers - 1)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through multi-layer causal liquid network

        Args:
            x: Input sequence (batch_size, seq_len, input_size)

        Returns:
            output: Predictions (batch_size, output_size)
            all_event_weights: Event weights from all layers
        """
        batch_size, seq_len, _ = x.size()

        # Initialize hidden states for all layers
        hidden_states = [
            torch.zeros(batch_size, self.hidden_size, device=x.device)
            for _ in range(self.num_layers)
        ]

        # Reset previous inputs for all causal cells (new sequence)
        for cell in self.liquid_cells:
            cell.prev_input = None

        # Track event weights from all layers
        all_event_weights = [[] for _ in range(self.num_layers)]

        # Process sequence with temporal causality
        for t in range(seq_len):
            x_t = x[:, t, :]

            # Process through causal liquid layers
            for layer_idx in range(self.num_layers):
                # Input to this layer
                if layer_idx == 0:
                    layer_input = x_t
                else:
                    layer_input = hidden_states[layer_idx - 1]

                # Update through causal liquid cell
                new_hidden, event_weight = self.liquid_cells[layer_idx](
                    layer_input, hidden_states[layer_idx]
                )

                # Apply layer normalization
                new_hidden = self.layer_norms[layer_idx](new_hidden)

                # Add skip connection (direct causal path)
                if layer_idx > 0:
                    skip = self.skip_connections[layer_idx - 1](hidden_states[layer_idx - 1])
                    new_hidden = new_hidden + skip

                hidden_states[layer_idx] = new_hidden
                all_event_weights[layer_idx].append(event_weight)

        # Use final layer's hidden state for prediction
        final_hidden = hidden_states[-1]

        # Generate output
        output = self.output_layer(final_hidden)

        # Stack event weights
        all_event_weights = [
            torch.cat(weights, dim=1) for weights in all_event_weights
        ]

        return output, all_event_weights


class CausalEventLoss(nn.Module):
    """
    Causal event-weighted loss function

    Weights prediction errors based on causal events (appliance state changes)
    to focus learning on causally important moments
    """

    def __init__(self, base_loss='mse', event_weight_scale=2.0):
        """
        Args:
            base_loss: Base loss function ('mse' or 'mae')
            event_weight_scale: Scale factor for event importance
        """
        super(CausalEventLoss, self).__init__()

        self.event_weight_scale = event_weight_scale

        if base_loss == 'mse':
            self.base_criterion = nn.MSELoss(reduction='none')
        elif base_loss == 'mae':
            self.base_criterion = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")

    def forward(self, predictions, targets, event_weights=None):
        """
        Compute causal event-weighted loss

        Args:
            predictions: Model predictions (batch_size, output_size)
            targets: Ground truth (batch_size, output_size)
            event_weights: Causal event weights (batch_size, seq_len) or None

        Returns:
            loss: Weighted loss value
        """
        # Base loss
        losses = self.base_criterion(predictions, targets)

        if event_weights is not None:
            # Average event weights across sequence
            avg_event_weight = event_weights.mean(dim=1, keepdim=True)

            # Weight loss by causal events
            # Higher weight when model detects causal events
            weights = 1.0 + self.event_weight_scale * avg_event_weight
            losses = losses * weights

        return losses.mean()


def detect_causal_events(y_true, threshold=10.0):
    """
    Detect causal events (appliance state changes) in ground truth

    Args:
        y_true: Ground truth appliance power (batch_size, seq_len)
        threshold: Threshold for detecting state changes

    Returns:
        event_mask: Binary mask of causal events (1 = event, 0 = no event)
    """
    if len(y_true.shape) == 1:
        y_true = y_true.unsqueeze(0)

    # Compute differences (state changes)
    differences = torch.abs(y_true[:, 1:] - y_true[:, :-1])

    # Detect events
    events = (differences > threshold).float()

    # Pad to match original length
    events = torch.cat([
        torch.zeros(y_true.size(0), 1, device=y_true.device),
        events
    ], dim=1)

    return events


def compute_granger_causality_score(x, y, max_lag=5):
    """
    Compute Granger causality score between two time series

    Tests if past values of x help predict y beyond y's own past

    Args:
        x: First time series (numpy array)
        y: Second time series (numpy array)
        max_lag: Maximum lag to consider

    Returns:
        score: Granger causality score (higher = stronger causality)
    """
    from sklearn.linear_model import LinearRegression

    n = len(y) - max_lag

    # Create lagged features
    X_restricted = np.column_stack([
        y[i:i+n] for i in range(max_lag)
    ])

    X_full = np.column_stack([
        np.column_stack([y[i:i+n] for i in range(max_lag)]),
        np.column_stack([x[i:i+n] for i in range(max_lag)])
    ])

    y_target = y[max_lag:]

    # Fit models
    model_restricted = LinearRegression().fit(X_restricted, y_target)
    model_full = LinearRegression().fit(X_full, y_target)

    # Compute RSS (Residual Sum of Squares)
    rss_restricted = np.sum((y_target - model_restricted.predict(X_restricted)) ** 2)
    rss_full = np.sum((y_target - model_full.predict(X_full)) ** 2)

    # Granger causality score
    score = (rss_restricted - rss_full) / rss_full if rss_full > 0 else 0

    return max(0, score)  # Return non-negative score


if __name__ == "__main__":
    print("Causal Liquid Neural Network Module")
    print("=" * 70)

    # Test basic causal liquid network
    print("\n1. Testing Causal Liquid Network Model")
    model = CausalLiquidNetworkModel(
        input_size=1,
        hidden_size=64,
        output_size=1,
        dt=0.1
    )

    # Create sample input
    batch_size = 4
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 1)

    # Forward pass
    output, event_weights = model(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Event weights shape: {event_weights.shape}")
    print(f"   Average event weight: {event_weights.mean().item():.4f}")

    # Test advanced causal liquid network
    print("\n2. Testing Advanced Causal Liquid Network Model")
    adv_model = AdvancedCausalLiquidNetworkModel(
        input_size=1,
        hidden_size=64,
        output_size=1,
        num_layers=3,
        dt=0.1
    )

    output, all_event_weights = adv_model(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Number of layers: {len(all_event_weights)}")
    for i, weights in enumerate(all_event_weights):
        print(f"   Layer {i+1} event weights - mean: {weights.mean().item():.4f}")

    # Test causal event loss
    print("\n3. Testing Causal Event Loss")
    loss_fn = CausalEventLoss(base_loss='mse', event_weight_scale=2.0)

    predictions = torch.randn(batch_size, 1)
    targets = torch.randn(batch_size, 1)

    loss_without_events = loss_fn(predictions, targets, event_weights=None)
    loss_with_events = loss_fn(predictions, targets, event_weights=event_weights)

    print(f"   Loss without events: {loss_without_events.item():.4f}")
    print(f"   Loss with events: {loss_with_events.item():.4f}")

    # Test causal event detection
    print("\n4. Testing Causal Event Detection")
    y_true = torch.cat([
        torch.zeros(50),
        torch.ones(30) * 100,
        torch.zeros(20)
    ]).unsqueeze(0)

    events = detect_causal_events(y_true, threshold=10.0)
    print(f"   True signal shape: {y_true.shape}")
    print(f"   Detected events shape: {events.shape}")
    print(f"   Number of events detected: {events.sum().item():.0f}")

    # Test Granger causality
    print("\n5. Testing Granger Causality")
    x_series = np.random.randn(200)
    y_series = np.random.randn(200)
    # Create weak causal relationship
    y_series[10:] += 0.3 * x_series[:-10]

    score = compute_granger_causality_score(x_series, y_series, max_lag=5)
    print(f"   Granger causality score: {score:.4f}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("\nKey Causal Features:")
    print("  ✓ Temporal causality constraints")
    print("  ✓ Causal event detection")
    print("  ✓ Event-weighted loss function")
    print("  ✓ Multi-layer causal processing")
    print("  ✓ Granger causality analysis")
