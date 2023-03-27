from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import Tensor
from tqdm import tqdm

from .custom_transe import CustomTransE


class TransEModule(pl.LightningModule):
    """
    LightningModule for training and evaluating TransE algorithm for knowledge graph embeddings.

    Attributes:
        model(CustomTransE): The TransE model.
        learning_rate (float): The learning rate to use for the optimizer.
    """

    def __init__(self, learning_rate: float, num_ents: int, num_rels: int) -> None:
        super().__init__()

        self.model = CustomTransE(num_ents=num_ents, num_rels=num_rels)
        self.learning_rate = learning_rate

    def forward(
        self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor
    ) -> Tensor:
        return self.model(head_index, rel_type, tail_index)

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        head_idxs, rel_idxs, tail_idxs = batch
        loss = self.model.loss(head_idxs, rel_idxs, tail_idxs)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def _metrics_step(
        self, batch: Tuple[Tensor, Tensor, Tensor]
    ) -> Dict[str, Union[Tensor, List[int], List[bool]]]:
        head_idxs, rel_idxs, tail_idxs = batch
        loss = self.model.loss(head_idxs, rel_idxs, tail_idxs)

        mean_ranks, hits_at_10 = [], []
        for h, r, t in tqdm(
            zip(head_idxs, rel_idxs, tail_idxs),
            total=len(head_idxs),
            desc="Iterating through batch",
            leave=False,
        ):
            scores = []
            tail_indices = torch.arange(self.model.num_ents, device=t.device)
            for ts in tail_indices.split(head_idxs.numel()):
                scores.append(self(h.expand_as(ts), r.expand_as(ts), ts))
            rank = int(
                (torch.cat(scores).argsort(descending=True) == t).nonzero().view(-1)
            )
            mean_ranks.append(rank)
            hits_at_10.append(rank < 10)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss, "mean_ranks": mean_ranks, "hits_at_10": hits_at_10}

    def _metrics_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        mean_ranks = [
            mean_rank for output in outputs for mean_rank in output["mean_ranks"]
        ]
        hits_at_10 = [
            hit_at_10 for output in outputs for hit_at_10 in output["hits_at_10"]
        ]
        mean_reciprocal_rank = sum(
            1.0 / (mean_rank + 1) for mean_rank in mean_ranks
        ) / len(mean_ranks)

        mean_rank_metric = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        hits_at_10_metric = sum(hits_at_10) / len(hits_at_10)

        self.log_dict(
            {
                "MR": mean_rank_metric,
                "MRR": mean_reciprocal_rank,
                "Hits@10": hits_at_10_metric,
            }
        )

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: Any):
        return self._metrics_step(batch)

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:  # type: ignore[override]
        self._metrics_epoch_end(outputs)

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: Any):
        return self._metrics_step(batch)

    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:  # type: ignore[override]
        self._metrics_epoch_end(outputs)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
