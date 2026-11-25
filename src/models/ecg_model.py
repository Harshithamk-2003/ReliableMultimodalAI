import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze and Excitation — learns which leads are most important"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        s = self.squeeze(x).view(b, c)
        e = self.excitation(s).view(b, c, 1)
        return x * e

class TCNBlock(nn.Module):
    """Temporal Convolutional Block with residual connection"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.se = SEBlock(out_ch)
        # Residual projection
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        out = self.se(out)
        # Trim to match residual length
        out = out[:, :, :res.shape[2]]
        return self.relu(out + res)

class ECGModel(nn.Module):
    """
    12-lead ECG classifier
    Input:  (batch, 12, 1000)
    Output: (batch, num_classes)
    """
    def __init__(self, num_classes=13, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate

        # TCN blocks with increasing dilation
        self.tcn1 = TCNBlock(12,  64,  kernel_size=7, dilation=1)
        self.tcn2 = TCNBlock(64,  128, kernel_size=7, dilation=2)
        self.tcn3 = TCNBlock(128, 256, kernel_size=7, dilation=4)
        self.tcn4 = TCNBlock(256, 256, kernel_size=7, dilation=8)

        # Global pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.tcn3(x)
        x = self.tcn4(x)
        # Combine average and max pooling
        avg = self.gap(x).squeeze(-1)
        mx  = self.gmp(x).squeeze(-1)
        x = torch.cat([avg, mx], dim=1)  # (batch, 512)
        return self.classifier(x)

    def predict_with_uncertainty(self, x, n_samples=30):
        """Monte Carlo Dropout for uncertainty estimation"""
        self.train()  # keep dropout active
        preds = torch.stack([
            torch.softmax(self.forward(x), dim=1)
            for _ in range(n_samples)
        ])
        mean = preds.mean(0)
        uncertainty = preds.std(0).mean(1)  # mean uncertainty across classes
        return mean, uncertainty