import pytorch_lightning as pl
from torch_geometric.datasets import WordNet18RR  # type: ignore[import]
from torch_geometric.nn.kge.loader import KGTripletLoader  # type: ignore[import]


class EmbeddingsDataModule(pl.LightningDataModule):
    """
    LightningDataModule for a knowledge graph WordNet18RR.

    Attributes:
        data_path (str): The path to the directory containing the knowledge graph data.
        batch_size (int): The size of the batch to be used for training, validation, and testing.
        data (WordNet18RR): An instance of the WordNet18RR class that loads the knowledge graph data.
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 20000,
    ) -> None:
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.data = WordNet18RR(self.data_path)

    @property
    def num_ents(self) -> int:
        """
        Returns the number of unique entities in the knowledge graph.
        """
        return int(self.data.edge_index.max()) + 1

    @property
    def num_rels(self) -> int:
        """
        Returns the number of unique relations in the knowledge graph.
        """
        return int(self.data.edge_type.max()) + 1

    def train_dataloader(self) -> KGTripletLoader:
        """
        Returns a KGTripletLoader instance that provides batches of training triplets from the knowledge graph.
        """
        return KGTripletLoader(
            head_index=self.data.edge_index[:, self.data.train_mask][0],
            rel_type=self.data.edge_type[self.data.train_mask],
            tail_index=self.data.edge_index[:, self.data.train_mask][1],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self) -> KGTripletLoader:
        """
        Returns a KGTripletLoader instance that provides batches of validation triplets from the knowledge graph.
        """
        return KGTripletLoader(
            head_index=self.data.edge_index[:, self.data.val_mask][0],
            rel_type=self.data.edge_type[self.data.val_mask],
            tail_index=self.data.edge_index[:, self.data.val_mask][1],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
        )

    def test_dataloader(self) -> KGTripletLoader:
        """
        Returns a KGTripletLoader instance that provides batches of test triplets from the knowledge graph.
        """
        return KGTripletLoader(
            head_index=self.data.edge_index[:, self.data.test_mask][0],
            rel_type=self.data.edge_type[self.data.test_mask],
            tail_index=self.data.edge_index[:, self.data.test_mask][1],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
        )
