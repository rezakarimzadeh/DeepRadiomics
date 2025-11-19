import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
from torch.optim.lr_scheduler import LambdaLR
from .utils import MaskedBatchNorm1d


class RadiomicsTransformer(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        config: dict,


    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.BN = MaskedBatchNorm1d(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, config['d_model']),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True,
            activation="relu",
            norm_first=False,
            bias=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=config['num_layers'])
        self.head = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']//2),
            nn.ReLU(),
            nn.Linear(config['d_model']//2, config['output_dim']),
        )
        self.criterion = nn.CrossEntropyLoss()
        # Metrics
        self.acc = BinaryAccuracy()
        self.auroc = BinaryAUROC()
        self.f1 = BinaryF1Score()

    def forward(self, x, pad_mask=None):
        """
        x: [B, T, F]
        pad_mask: [B, T] bool, True = PAD (ignored)
        """
        B, T, F = x.shape
        device = x.device
        x = self.BN(x, pad_mask=pad_mask)  # [B, T, F]
        x = self.proj(x)                # [B, T, d_model]
        
        # attention mask (batch_first=True)
        x = self.transformer(x, src_key_padding_mask=pad_mask)  # [B, T, d_model]

        # masked mean pooling
        if pad_mask is not None:
            valid = (~pad_mask).float()                   
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1e-6)
            x = (x * valid.unsqueeze(-1)).sum(dim=1) / denom   
        else:
            x = x.mean(dim=1)

        logits = self.head(x).squeeze(1)                   
        return logits

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

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        total_epochs = self.config['max_epochs']
        warmup_epochs = self.config['warmup_epochs']

        # Cosine decay with linear warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cosine_decay

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda=lr_lambda),
            "interval": "epoch",
            "frequency": 1,
            "name": "cosine_warmup_lr"
        }

        return [optimizer], [scheduler]
