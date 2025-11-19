from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
from .utils import MaskedBatchNorm1d
from torch.optim.lr_scheduler import LambdaLR

def mlp(sizes, act=nn.ReLU, dropout=0.0, batchnorm=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if batchnorm:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
        if i < len(sizes) - 2:  # no act/dropout on last layer
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def masked_pool(x, mask, mode: Literal["sum", "mean", "max"] = "sum"):
    """
    x:    [B, T, D]
    mask: [B, T] with True for VALID elements (False for pad) OR None
    """
    if mask is None:
        if mode == "sum":
            return x.sum(dim=1)
        elif mode == "mean":
            return x.mean(dim=1)
        elif mode == "max":
            return x.max(dim=1).values
        else:
            raise ValueError("pool mode must be sum/mean/max")

    # expand mask to [B, T, 1]
    m = mask.unsqueeze(-1).to(x.dtype)
    if mode == "sum":
        return (x * m).sum(dim=1)
    elif mode == "mean":
        denom = m.sum(dim=1).clamp_min(1.0)  # avoid div by zero
        return (x * m).sum(dim=1) / denom
    elif mode == "max":
        # put -inf where masked to ignore in max
        neg_inf = torch.finfo(x.dtype).min
        x_masked = x.masked_fill(~mask.unsqueeze(-1), neg_inf)
        return x_masked.max(dim=1).values
    else:
        raise ValueError("pool mode must be sum/mean/max")


class DeepSetsInvariant(nn.Module):
    """
    f(X) = rho( Pool_{x in X} phi(x) )

    Inputs:
      - x:    [B, T, F]  (padded to same T across batch)
      - mask: [B, T] boolean, False for VALID elements (True for padding)

    Returns:
      - logits: [B, C]
    """
    def __init__(
        self,
        in_dim: int,
        phi_hidden: int = 128,
        rep_dim: int = 128,
        rho_hidden: int = 128,
        num_classes: int = 2,
        pool: Literal["sum", "mean", "max"] = "sum",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pool = pool

        self.BN_layer = MaskedBatchNorm1d(in_dim)
        # phi: per-element embedding
        self.phi = mlp([in_dim, phi_hidden, rep_dim], dropout=dropout)

        # rho: set-level head
        self.rho = mlp([rep_dim, rho_hidden, num_classes], dropout=dropout, batchnorm=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x:[B,T,F], mask:[B,T] False=valid
        x = self.BN_layer(x, mask)
        z = self.phi(x)                       # [B,T,rep_dim]
        z_pool = masked_pool(z, ~mask, self.pool)  # [B,rep_dim]
        logits = self.rho(z_pool)             # [B,C]
        return logits
    

class RadiomicsDeepSets(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        config: dict
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.lr = config['lr']
        self.max_epochs = config['max_epochs']

        self.model = DeepSetsInvariant(
            in_dim=input_dim,
            phi_hidden=config['phi_hidden'],
            rep_dim=config['rep_dim'],
            rho_hidden=config['rho_hidden'],
            num_classes=config['num_classes'],
            pool=config['pool'],
            dropout=config['dropout'],
        )
        self.criterion = nn.CrossEntropyLoss()
        # Metrics
        self.acc = BinaryAccuracy()
        self.auroc = BinaryAUROC()
        self.f1 = BinaryF1Score()
        
    def forward(self, x, mask=None):
        return self.model(x, mask)

    def _shared_step(self, batch, stage):
        x, y, pad_mask = batch['features'], batch['labels'], batch.get('pad_mask', None)
        logits = self(x, mask=pad_mask)
        loss = self.criterion(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_acc", self.acc(y_hat, y.int()), prog_bar=True)
        self.log(f"{stage}_auroc", self.auroc(y_hat, y.int()), prog_bar=False)
        self.log(f"{stage}_f1", self.f1(y_hat, y.int()), prog_bar=False)
        return loss

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