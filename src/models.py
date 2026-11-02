"""Deep models: CNN, BiLSTM, Transformer and the proposed hybrid CNN-BiLSTM-Transformer (PyTorch)."""
import math
import torch
import torch.nn as nn
import config


def conv_block(cin, cout, k=7):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU(),
                         nn.MaxPool1d(2), nn.Dropout(0.1))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):                      # x: (B, T, d)
        return x + self.pe[:, : x.size(1)]


def transformer_encoder(d_model, nhead, layers, dropout):
    layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 2, dropout=dropout,
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers)


class CNNOnly(nn.Module):
    def __init__(self, n_leads=config.N_LEADS, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(conv_block(n_leads, 32), conv_block(32, 64), conv_block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class BiLSTMOnly(nn.Module):
    def __init__(self, n_leads=config.N_LEADS, n_classes=2, hidden=64):
        super().__init__()
        self.down = nn.AvgPool1d(4)                       # 1000 -> 250 time steps (keeps LSTM fast)
        self.lstm = nn.LSTM(n_leads, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden * 2, n_classes))

    def forward(self, x):
        h, _ = self.lstm(self.down(x).transpose(1, 2))
        return self.head(h.mean(dim=1))


class TransformerOnly(nn.Module):
    def __init__(self, n_leads=config.N_LEADS, n_classes=2, d_model=128, nhead=4, layers=2):
        super().__init__()
        self.embed = nn.Conv1d(n_leads, d_model, kernel_size=8, stride=8)   # 1000 -> 125 tokens
        self.pos = PositionalEncoding(d_model)
        self.enc = transformer_encoder(d_model, nhead, layers, 0.1)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.3), nn.Linear(d_model, n_classes))

    def forward(self, x):
        z = self.pos(self.embed(x).transpose(1, 2))
        return self.head(self.enc(z).mean(dim=1))


class HybridCNNBiLSTMTransformer(nn.Module):
    """CNN (local morphology) -> BiLSTM (bidirectional temporal context) -> Transformer (self-attention)
    -> global pooling -> fully connected layers -> softmax (applied in the loss / at inference)."""

    def __init__(self, n_leads=config.N_LEADS, n_classes=2, lstm_hidden=64, d_model=128, nhead=4, layers=2):
        super().__init__()
        self.cnn = nn.Sequential(conv_block(n_leads, 32), conv_block(32, 64), conv_block(64, 128))   # 1000 -> 125
        self.bilstm = nn.LSTM(128, lstm_hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(lstm_hidden * 2, d_model)
        self.pos = PositionalEncoding(d_model)
        self.enc = transformer_encoder(d_model, nhead, layers, 0.1)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.3), nn.Linear(d_model, 64), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(64, n_classes))

    def forward(self, x):                                   # x: (B, 12, 1000)
        f = self.cnn(x).transpose(1, 2)                     # (B, 125, 128)
        h, _ = self.bilstm(f)                               # (B, 125, 2*hidden)
        z = self.enc(self.pos(self.proj(h)))                # (B, 125, d_model)
        return self.head(z.mean(dim=1))


def build_model(name):
    return {"CNN": CNNOnly, "BiLSTM": BiLSTMOnly, "Transformer": TransformerOnly,
            "CNN-BiLSTM-Transformer": HybridCNNBiLSTMTransformer}[name]()
