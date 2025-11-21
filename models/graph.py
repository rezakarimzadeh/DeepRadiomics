import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import Data
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
from .utils import MaskedBatchNorm1d


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):

        super(GCN, self).__init__()
        _gcn = GATConv
        # A list of GCNConv layers
        self.convs = torch.nn.ModuleList([_gcn(in_channels=input_dim, out_channels=hidden_dim)])
        if num_layers > 2:
            self.convs.extend([_gcn(in_channels=hidden_dim, out_channels=hidden_dim) for i in range(num_layers-2)])
        self.convs.extend([_gcn(in_channels=hidden_dim, out_channels=output_dim)])
        # A list of 1D batch normalization layers
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for i in range(num_layers-1)])

        # The log softmax layer
        self.softmax = torch.nn.LogSoftmax()

        # Probability of an element to be zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_weight):
        for gcn, bn in zip(self.convs[:-1], self.bns):
            x = gcn(x, adj_t, edge_weight)
            # x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        # last layer
        x = self.convs[-1](x, adj_t, edge_weight)

        if self.return_embeds:
            out = x
        else:
            out = self.softmax(x)

        return out


    
### GCN to predict graph property
class GCN_Graph(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()

        # Node embedding model
        self.BN = torch.nn.BatchNorm1d(input_dim)
        
        self.gnn_node = GCN(input_dim, hidden_dim,
            hidden_dim, num_layers, dropout, return_embeds=True)

        self.pool = global_mean_pool

        # Output layer
        self.linear = nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim//2),
                                    # torch.nn.BatchNorm1d(hidden_dim//2),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim//2, output_dim)
        )


    def reset_parameters(self):
      self.gnn_node.reset_parameters()
      self.linear.reset_parameters()

    def forward(self, data):
        x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_attr
        x = self.BN(x)
        out = self.gnn_node(x, edge_index, edge_weight)
        out = self.pool(out, batch)
        # add last feature vector into the gnn outputs
        # last_point_features = self.pool(x, batch)
        # classifier_input = torch.cat([last_point_features, out], dim=1)

        out = self.linear(out)
        return torch.squeeze(out)


    
class RadiomicsGraph(pl.LightningModule):
    def __init__(self, input_dim: int, config: dict):
        super(RadiomicsGraph, self).__init__()
        self.save_hyperparameters()
        self.model = GCN_Graph(input_dim, config['hidden_dim'], config['num_classes'], config['num_layers'], config['dropout'])
        self.lr = config['lr']
        self.max_epochs = config['max_epochs']
        self.criterion = nn.CrossEntropyLoss()
        # Metrics
        self.acc = BinaryAccuracy()
        self.auroc = BinaryAUROC()
        self.f1 = BinaryF1Score()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        logits = self(batch)
        y = batch.y.long()
        loss = self.criterion(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch.num_graphs)
        self.log(f"{stage}_acc", self.acc(y_hat, y.int()), prog_bar=True, batch_size=batch.num_graphs)
        self.log(f"{stage}_auroc", self.auroc(y_hat, y.int()), prog_bar=False, batch_size=batch.num_graphs)
        self.log(f"{stage}_f1", self.f1(y_hat, y.int()), prog_bar=False, batch_size=batch.num_graphs)
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




