import math
from typing import Any, List, Optional

import torch
import torch.nn as nn

from examples.nbody.networks.gcl import GCL


class BaseEuclideangraphModel(nn.Module):
    def __init__(
        self,
        hyperparams: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.learning_rate: Optional[float] = (
            hyperparams.learning_rate if hasattr(hyperparams, "learning_rate") else None
        )
        self.weight_decay: float = (
            hyperparams.weight_decay if hasattr(hyperparams, "weight_decay") else 0.0
        )
        self.patience: int = (
            hyperparams.patience if hasattr(hyperparams, "patience") else 100
        )


class GNN(BaseEuclideangraphModel):
    def __init__(self, hyperparams: Any) -> None:
        super().__init__(hyperparams)
        self.model: str = "GNN"
        self.hidden_dim: int = hyperparams.hidden_dim
        self.input_dim: int = hyperparams.input_dim
        self.n_layers: int = hyperparams.num_layers
        self.act_fn: nn.Module = nn.SiLU()
        self.attention: bool = False
        self.recurrent: bool = True

        for i in range(0, self.n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    input_nf=self.hidden_dim,
                    output_nf=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    edges_in_nf=2,
                    act_fn=self.act_fn,
                    attention=self.attention,
                    recurrent=self.recurrent,
                ),
            )

        self.decoder: nn.Module = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.act_fn,
            nn.Linear(self.hidden_dim, 3),
        )
        self.embedding: nn.Module = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim)
        )

    def forward(
        self,
        nodes: torch.Tensor,
        loc: torch.Tensor,
        edges: List[torch.Tensor],
        vel: torch.Tensor,
        edge_attr: torch.Tensor,
        _: Any,
    ) -> torch.Tensor:
        nodes = torch.cat([loc, vel], dim=1)
        h = self.embedding(nodes)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
        return self.decoder(h)


class Transformer(BaseEuclideangraphModel):
    def __init__(self, hyperparams: Any) -> None:
        super().__init__(hyperparams)
        self.model: str = "Transformer"
        self.hidden_dim: int = hyperparams.hidden_dim
        self.input_dim: int = hyperparams.input_dim
        self.n_layers: int = hyperparams.num_layers
        self.ff_hidden: int = hyperparams.ff_hidden
        self.act_fn: nn.Module = nn.ReLU()
        self.dropout: float = 0
        self.nhead: int = hyperparams.nheads

        self.pos_encoder: nn.Module = PositionalEncoding(
            hidden_dim=self.hidden_dim, dropout=self.dropout
        )

        self.charge_embedding: nn.Module = nn.Embedding(2, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=7 * self.hidden_dim,
            nhead=self.nhead,
            dim_feedforward=self.ff_hidden,
            batch_first=True,
        )
        self.encoder: nn.Module = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=self.n_layers
        )

        self.decoder: nn.Module = nn.Sequential(
            nn.Linear(
                in_features=7 * self.hidden_dim, out_features=7 * self.hidden_dim
            ),
            self.act_fn,
            nn.Linear(in_features=7 * self.hidden_dim, out_features=3),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        loc: torch.Tensor,
        edges: List[torch.Tensor],
        vel: torch.Tensor,
        edge_attr: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        pos_encodings = torch.cat([loc, vel], dim=1).unsqueeze(2)
        pos_encodings = self.pos_encoder(pos_encodings)
        charges[charges == -1] = 0
        charges = charges.long()
        charges = self.charge_embedding(charges)
        nodes = torch.cat([pos_encodings, charges], dim=1)
        nodes = nodes.view(-1, 5, nodes.shape[1] * nodes.shape[2])
        h = self.encoder(nodes)
        h = h.view(-1, h.shape[2])
        h = self.decoder(h)
        return h


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = hidden_dim
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        ).view(1, 1, int(hidden_dim / 2))
        self.register_buffer("div_term", div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = torch.zeros(x.shape[0], x.shape[1], self.hidden_dim).to(x.device)
        sin_terms = torch.sin(x * self.div_term)
        pe[:, :, 0::2] = sin_terms
        pe[:, :, 1::2] = torch.cos(x * self.div_term)
        return self.dropout(pe)
