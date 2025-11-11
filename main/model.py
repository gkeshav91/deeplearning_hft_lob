#!/opt/python/3.10/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
#################################################

# positional excoding 
def get_sinusoidal_encoding(seq_len, dim, device='cpu'):
    """Generates [seq_len, dim] sinusoidal positional encodings."""
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-torch.log(torch.tensor(10000.0)) / dim))
    
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [seq_len, dim]

#################################################

# multi-head transformer module 
class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, Dp, Tin, num_heads, dropout):
        super().__init__()
        assert Dp % num_heads == 0, "Dp must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = Dp // num_heads

        self.q_proj = nn.Linear(Dp, Dp)
        self.k_proj = nn.Linear(Dp, Dp)
        self.v_proj = nn.Linear(Dp, Dp)
        self.out_proj = nn.Linear(Dp, Dp)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(Tin, Tin)).unsqueeze(0))  # [1, Tin, Tin]

    def forward(self, x):
        # x: [B, Dp, Tin]
        B, Dp, Tin = x.size()
        x = x.transpose(1, 2)  # [B, Tin, Dp]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, Tin, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Tin, Dh]
        K = K.view(B, Tin, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Tin, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)  # [B, H, Tin, Tin]
        attn_scores = attn_scores.masked_fill(self.mask[:, :Tin, :Tin] == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = attn_weights @ V  # [B, H, Tin, Dh]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Tin, Dp)  # [B, Tin, Dp]
        out = self.out_proj(attn_output)  # [B, Tin, Dp]

        return out.transpose(1, 2)  # [B, Dp, Tin]

#################################################

class EnhancedTABL(nn.Module):
    def __init__(self, Din, Tin, Dp, Tp, num_heads, dropout):
        super().__init__()
        self.Din = Din
        self.Tin = Tin
        self.Dp = Dp

        self.feature_proj = nn.Linear(Din, Dp)

        # register_buffer so it's moved with .to() but not trainable
        pe = get_sinusoidal_encoding(Tin, Dp)
        self.register_buffer("pos_encoding", pe.unsqueeze(0))  # [1, Tin, Dp]

        self.attn = MultiHeadTemporalAttention(Dp, Tin, num_heads, dropout)

        self.lam = nn.Parameter(torch.zeros(Dp, 1))
        self.norm = nn.LayerNorm([Dp, Tin])

        self.temporal_proj = nn.Linear(Tin, Tp)
        self.bias = nn.Parameter(torch.zeros(Dp, Tp))

    def forward(self, x):
        # x: [B, Din, Tin]
        B, Din, Tin = x.shape
        assert Tin == self.Tin, f"Expected Tin={self.Tin}, got {Tin}"

        # Feature projection
        x_proj = self.feature_proj(x.transpose(1, 2))  # [B, Tin, Dp]
        x_proj = x_proj + self.pos_encoding[:, :Tin, :]  # Add fixed positional encoding
        x_proj = x_proj.transpose(1, 2)  # [B, Dp, Tin]

        # Attention
        A = self.attn(x_proj)  # [B, Dp, Tin]

        # Gated combination
        lam = torch.sigmoid(self.lam).unsqueeze(0)  # [1, Dp, 1]
        Xt = lam * A + (1 - lam) * x_proj

        # Residual + LayerNorm
        Xt = self.norm(Xt + x_proj)

        # Final temporal projection
        out = self.temporal_proj(Xt) + self.bias  # [B, Dp, Tp]
        return F.relu(out)

#################################################

class BilinearLayer(nn.Module):
    def __init__(self, Din, Tin, Dout):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(Dout, Din))
        self.B  = nn.Parameter(torch.zeros(Dout, Tin))
        nn.init.xavier_uniform_(self.W1);
    def forward(self, x): return F.relu(self.W1 @ x + self.B)

#################################################

class ConvTABLNet(nn.Module):
    HIDDEN = {
        'A': [],
        'B1': [(40*3)],
        'B2': [(15*3)],
        'C': [(40*3),(40)]
    }
    def __init__(self, variant, D, T, Dp, Tp, p_drop, num_heads):
        super().__init__()
        Dcur, Tcur = D, T
        self.hid = nn.ModuleList()
        for hD in self.HIDDEN[variant]:
            self.hid.append(BilinearLayer(Dcur, Tcur, hD))
            Dcur = hD
        self.tabl = EnhancedTABL(Dcur, Tcur, Dp, Tp, num_heads, p_drop)
        self.drop = nn.Dropout(p_drop)
        self.head = nn.Linear(Dp * Tp, 1)  # +2 for SK, OBI
    def forward(self, x):
        for layer in self.hid:
            x = layer(x)
        x = self.tabl(x).flatten(1)
        x = self.drop(x)
        return self.head(x).squeeze(-1)
    
########################################################


class DeepLOBRegression(nn.Module):
    def __init__(self,dropout_rate):
        super(DeepLOBRegression, self).__init__()

        # ---- Convolutional Blocks ----
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2))
        self.dropout1 = nn.Dropout2d(dropout_rate)  # Dropout with probability 0.2
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(2, 0))
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(2, 0))
        self.dropout3 = nn.Dropout2d(dropout_rate)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=(1, 2), stride=(1, 2))
        self.dropout4 = nn.Dropout2d(dropout_rate)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(2, 0))
        self.dropout5 = nn.Dropout2d(dropout_rate)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(2, 0))
        self.dropout6 = nn.Dropout2d(dropout_rate)

        self.conv7 = nn.Conv2d(16, 16, kernel_size=(1, 10))
        self.dropout7 = nn.Dropout2d(dropout_rate)
        self.conv8 = nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(2, 0))
        self.dropout8 = nn.Dropout2d(dropout_rate)
        self.conv9 = nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(2, 0))
        self.dropout9 = nn.Dropout2d(dropout_rate)

        # Inception module
        self.branch1x1_1 = nn.Conv2d(16, 32, kernel_size=(1, 1))
        self.branch1x1_3 = nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0))

        self.branch1x1_2 = nn.Conv2d(16, 32, kernel_size=(1, 1))
        self.branch1x1_5 = nn.Conv2d(32, 32, kernel_size=(5, 1), padding=(2, 0))

        self.branch_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.branch_pool_conv = nn.Conv2d(16, 32, kernel_size=(1, 1))

        # LSTM and regression head
        self.lstm = nn.LSTM(input_size=96, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)  # single output for regression

    def forward(self, x):
        # x: (batch, 1, 100, 40)
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)  # Apply dropout after ReLU
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)

        x = F.relu(self.conv4(x))
        x = self.dropout4(x)
        x = F.relu(self.conv5(x))
        x = self.dropout5(x)
        x = F.relu(self.conv6(x))
        x = self.dropout6(x)

        x = F.relu(self.conv7(x))
        x = self.dropout7(x)
        x = F.relu(self.conv8(x))
        x = self.dropout8(x)
        x = F.relu(self.conv9(x))
        x = self.dropout9(x)

        # Inception branches
        b1 = F.relu(self.branch1x1_1(x))
        b1 = F.relu(self.branch1x1_3(b1))

        b2 = F.relu(self.branch1x1_2(x))
        b2 = F.relu(self.branch1x1_5(b2))

        b3 = self.branch_pool(x)
        b3 = F.relu(self.branch_pool_conv(b3))

        # Concatenate along channel axis
        x = torch.cat([b1, b2, b3], dim=1)  # -> (batch, 96, 100, 1)
        x = x.squeeze(3)                   # -> (batch, 96, 100)
        x = x.permute(0, 2, 1)             # -> (batch, 100, 96)

        # LSTM
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Last time step only
        return out.squeeze(1)              # -> (batch,)
    

#################################################

class TABLDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray,num_features,num_time_steps ):
        """
        Args:
            X: shape (num_samples,num_features, num_time_steps)
            y: shape (num_samples,)
        """
        assert X.shape[1:] == (num_features,num_time_steps), "Input shape must be (num_features,num_time_steps)"
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])  # shape: (D, T)
        y = torch.tensor(self.y[idx])
        return x, y

class LOBRegressionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray,num_time_steps,num_features):
        """
        Args:
            X: shape (num_samples, num_time_steps, num_features)
            y: shape (num_samples,)
        """
        assert X.shape[1:] == (num_time_steps, num_features), "Input shape must be (num_time_steps, num_features)"
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Add channel dim: (1, num_time_steps, num_features)
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)
        y = torch.tensor(self.y[idx])
        return x, y

#################################################
