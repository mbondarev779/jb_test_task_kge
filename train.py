import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from src import EmbeddingsDataModule, TransEModule


def main():
    pl.seed_everything(42)

    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 20000
    DATA_PATH = "data"
    N_EPOCHS = 1000
    LR = 0.01

    data_module = EmbeddingsDataModule(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
    )

    model = TransEModule(
        learning_rate=LR, num_ents=data_module.num_ents, num_rels=data_module.num_rels
    )

    trainer = pl.Trainer(
        logger=WandbLogger(project="transe", log_model="all"),
        max_epochs=N_EPOCHS,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=False,
        accelerator=DEVICE,
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
