import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
from .utils import MaskedBatchNorm1d

class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionMIL, self).__init__()
        self.M = 128
        self.L = 128
        self.ATTENTION_BRANCHES = 1
        
        self.BN = MaskedBatchNorm1d(input_dim)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.M)
        )
        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, self.M//2),
            nn.BatchNorm1d(self.M//2),
            nn.ReLU(),
            nn.Linear(self.M//2, output_dim)
        )

    def forward(self, x, pad_mask, attention=False):
        x = self.BN(x, pad_mask)
        # x: [B, T, F]
        H = self.feature_extractor(x)  # [B, T, H]
        A = self.attention(H).squeeze(-1)  # [B, T]
        # Apply mask before softmax
        A = A.masked_fill(pad_mask, float('-inf'))
        A = F.softmax(A, dim=1)  # [B, T]
        M = torch.bmm(A.unsqueeze(1), H).squeeze(1)  # [B, H]
        M = M.view(M.size(0), -1)  # Flatten to [B, H]
        out = self.classifier(M)  # [B, C]
        if attention:
            return out, A
        return out

class GatedAttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GatedAttentionMIL, self).__init__()
        self.M = 128
        self.L = 64
        self.ATTENTION_BRANCHES = 1
        
        self.BN = MaskedBatchNorm1d(input_dim)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.M)
        )
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES)
        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, self.M//2),
            nn.BatchNorm1d(self.M//2),
            nn.ReLU(),
            nn.Linear(self.M//2, output_dim)
        )

    def forward(self, x, pad_mask):
        x = self.BN(x, pad_mask)
        # x: [B, T, F]
        H = self.feature_extractor(x)  # [B, T, H]
        A_V = self.attention_V(H)      # [B, T, L]
        A_U = self.attention_U(H)      # [B, T, L]
        A = self.attention_w(A_V * A_U).squeeze(-1)  # [B, T]
        # Apply mask before softmax
        A = A.masked_fill(pad_mask, float('-inf'))
        A = F.softmax(A, dim=1)  # [B, T]
        M = torch.bmm(A.unsqueeze(1), H).squeeze(1)  # [B, H]
        M = M.view(M.size(0), -1)  # Flatten to [B, H]
        out = self.classifier(M)  # [B, C]
        return out
    
class RadiomicsMIL(pl.LightningModule):
    def __init__(self, input_dim: int, config: dict):
        super(RadiomicsMIL, self).__init__()
        self.save_hyperparameters()
        # ========================= you can choose different MIL models here =========================
        self.model = AttentionMIL(input_dim, config['hidden_dim'], config['num_classes'])
        # self.model = GatedAttentionMIL(input_dim, config['hidden_dim'], config['num_classes'])
        
        self.lr = config['lr']
        self.max_epochs = config['max_epochs']
        self.criterion = nn.CrossEntropyLoss()
        # Metrics
        self.acc = BinaryAccuracy()
        self.auroc = BinaryAUROC()
        self.f1 = BinaryF1Score()

    def forward(self, x, pad_mask):
        return self.model(x, pad_mask)

    def _shared_step(self, batch, stage):
        x, y, pad_mask = batch['features'], batch['labels'], batch.get('pad_mask', None)
        logits = self(x, pad_mask=pad_mask)
        loss = self.criterion(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_acc", self.acc(y_hat, y.int()), prog_bar=True)
        self.log(f"{stage}_auroc", self.auroc(y_hat, y.int()), prog_bar=False)
        self.log(f"{stage}_f1", self.f1(y_hat, y.int()), prog_bar=False)
        return loss
    
    def get_pred_and_attention(self, x, pad_mask):
        logits, attention_weights = self.model(x, pad_mask=pad_mask, attention=True)
        return logits, attention_weights
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (self.lr/self.max_epochs)*(self.max_epochs - epoch) if epoch < self.max_epochs else 0)
        return [optimizer], [scheduler]