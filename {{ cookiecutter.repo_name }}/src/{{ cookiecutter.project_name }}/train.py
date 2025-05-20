from model import MyModel
from data_module import MyDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pathlib import Path
import torch

# 1. Load data
datamodule = MyDataModule(data_path=Path("data/data.csv"), batch_size=32)
datamodule.setup()

# 2. Infer input/output sizes
x, y = next(iter(datamodule.train_dataloader()))
input_dim = x.shape[1]
output_dim = len(torch.unique(y))

# 3. Init model
model = MyModel(input_dim, output_dim)

# 4. Define callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=1,
    mode="min",
    filename="best-checkpoint",
)

# 5. Train with callback
trainer = Trainer(
    max_epochs=10,
    callbacks=[checkpoint_callback]
)
trainer.fit(model, datamodule=datamodule)
