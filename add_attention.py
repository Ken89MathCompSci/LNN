# Temporary script to add attention models

with open('Source Code/models.py', 'r') as f:
    content = f.read()

attention_code = """

# ============================================================================
# ATTENTION-ENHANCED LIQUID NEURAL NETWORKS
# ============================================================================

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)

        return output, attn_weights


class AttentionLiquidNetworkModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt=0.1, num_heads=4, dropout=0.1):
        super(AttentionLiquidNetworkModel, self).__init__()
        self.hidden_size = hidden_size

        self.liquid_layer = LiquidTimeLayer(input_size, hidden_size, dt)
        self.attention = SelfAttention(hidden_size, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        hidden = None
        hidden_states = []

        for t in range(seq_length):
            x_t = x[:, t, :]
            hidden = self.liquid_layer(x_t, hidden)
            hidden_states.append(hidden)

        hidden_sequence = torch.stack(hidden_states, dim=1)
        attn_output, _ = self.attention(hidden_sequence)
        hidden_sequence = self.layer_norm(hidden_sequence + attn_output)

        final_hidden = hidden_sequence[:, -1, :]
        final_hidden = self.dropout(final_hidden)
        output = self.output_layer(final_hidden)

        return output


class AdvancedAttentionLiquidNetworkModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dt=0.1,
                 num_heads=4, dropout=0.1):
        super(AdvancedAttentionLiquidNetworkModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.liquid_layers = nn.ModuleList([
            AdvancedLiquidTimeLayer(
                input_size if i == 0 else hidden_size,
                hidden_size,
                dt
            ) for i in range(num_layers)
        ])

        self.attention_layers = nn.ModuleList([
            SelfAttention(hidden_size, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])

        self.skip_connections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers - 1)
        ]) if num_layers > 1 else None

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        hidden_states = [None] * self.num_layers
        layer_outputs = []

        for layer_idx in range(self.num_layers):
            layer_hidden_sequence = []

            for t in range(seq_length):
                if layer_idx == 0:
                    x_t = x[:, t, :]
                else:
                    x_t = layer_outputs[layer_idx - 1][:, t, :]

                hidden_states[layer_idx] = self.liquid_layers[layer_idx](
                    x_t, hidden_states[layer_idx]
                )
                layer_hidden_sequence.append(hidden_states[layer_idx])

            layer_sequence = torch.stack(layer_hidden_sequence, dim=1)
            attn_output, _ = self.attention_layers[layer_idx](layer_sequence)

            if layer_idx > 0 and self.skip_connections is not None:
                skip = self.skip_connections[layer_idx - 1](layer_outputs[layer_idx - 1])
                layer_sequence = layer_sequence + skip

            layer_sequence = self.layer_norms[layer_idx](layer_sequence + attn_output)
            layer_sequence = self.dropout_layers[layer_idx](layer_sequence)

            layer_outputs.append(layer_sequence)

        final_hidden = layer_outputs[-1][:, -1, :]
        output = self.output_layer(final_hidden)

        return output
"""

with open('Source Code/models.py', 'w') as f:
    f.write(content + attention_code)

print("SUCCESS")
