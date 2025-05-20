from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch


class MyDataset(Dataset):
    """Custom dataset that loads data from a CSV file."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.data = pd.read_csv(data_path)

        # Separate features and labels; assumes last column is label
        self.features = self.data.iloc[:, :-1].values
        self.labels = self.data.iloc[:, -1].values

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        x = torch.tensor(self.features[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess raw data (example: drop NA) and save to output."""
        output_folder.mkdir(parents=True, exist_ok=True)
        processed = self.data.dropna()
        processed.to_csv(output_folder / "processed.csv", index=False)


class MyDataModule(LightningDataModule):
    """LightningDataModule to wrap the dataset and provide DataLoaders."""

    def __init__(self, data_path: Path, batch_size=32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = MyDataset(self.data_path)

        # 80/20 train/val split
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        self.train_set, self.val_set = random_split(dataset, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)


# Optional: test usage
if __name__ == "__main__":
    path = Path("data/data.csv")
    dataset = MyDataset(path)
    print(len(dataset), dataset[0])
    
    datamodule = MyDataModule(path)
    datamodule.setup()
    for batch in datamodule.train_dataloader():
        print(batch)
        break
