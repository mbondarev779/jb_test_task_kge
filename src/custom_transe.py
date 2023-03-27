import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding


class CustomTransE(nn.Module):
    """
    Implementation of the TransE model for knowledge graph embedding.

    Attributes:
        num_ents (int): The number of entities in the knowledge graph.
        num_rels (int): The number of relations in the knowledge graph.
        margin (float): The margin used in the margin ranking loss function. Default is 1.0.
        norm (float): The p-norm used for normalization. Default is 1.0.
        dim (int): The dimensionality of the embedding space. Default is 50.
        ent_emb (nn.Embedding): The entity embedding matrix.
        rel_emb (nn.Embedding): The relation embedding matrix.
    """

    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        margin: float = 1.0,
        norm: float = 1.0,
        dim: int = 50,
    ) -> None:
        super(CustomTransE, self).__init__()

        self.num_ents = num_ents
        self.num_rels = num_rels
        self.margin = margin
        self.norm = norm
        self.dim = dim

        self.ent_emb = Embedding(self.num_ents, self.dim)
        self.rel_emb = Embedding(self.num_rels, self.dim)

        bound = 6.0 / math.sqrt(self.dim)
        self.ent_emb.weight.data.uniform_(-bound, bound)
        self.rel_emb.weight.data.uniform_(-bound, bound)

        F.normalize(
            input=self.rel_emb.weight.data,
            p=self.norm,
            dim=-1,
            out=self.rel_emb.weight.data,
        )

    def forward(self, head_idxs: Tensor, rel_idxs: Tensor, tail_idxs: Tensor) -> Tensor:
        """
        Computes the scores for triples composed of head entities, relations, and tail entities.
        """

        head = self.ent_emb(head_idxs)
        rel = self.rel_emb(rel_idxs)
        tail = self.ent_emb(tail_idxs)

        head = F.normalize(head, p=self.norm, dim=-1)
        tail = F.normalize(tail, p=self.norm, dim=-1)

        return -((head + rel) - tail).norm(p=self.norm, dim=-1)

    def loss(self, head_idxs: Tensor, rel_idxs: Tensor, tail_idxs: Tensor) -> Tensor:
        """
        Computes the margin ranking loss for positive and negative triples.
        """
        pos_score = self(head_idxs, rel_idxs, tail_idxs)
        with torch.no_grad():
            neg_head_idxs, neg_tail_idxs = self._random_sample(head_idxs, tail_idxs)
        neg_score = self(neg_head_idxs, rel_idxs, neg_tail_idxs)
        # поменять местами возможно
        return F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )

    def _random_sample(
        self, head_idxs: Tensor, tail_idxs: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Generates negative triples by replacing either the head or tail entities with random entities.
        """
        num_negatives = head_idxs.numel() // 2

        rnd_idxs = torch.randint(
            self.num_ents, head_idxs.size(), device=head_idxs.device
        )

        # for the first half, we replace heads with random entities
        head_idxs = head_idxs.clone()
        head_idxs[:num_negatives] = rnd_idxs[:num_negatives]

        # for the second half, we replace tails with random entities
        tail_idxs = tail_idxs.clone()
        tail_idxs[num_negatives:] = rnd_idxs[num_negatives:]

        return head_idxs, tail_idxs
