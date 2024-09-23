import pandas as pd
from torch.utils.data import Dataset
import random
import torch

class StockDataset(Dataset):
    def __init__(self, data_dir="./src/stock_data/all_stocks_5yr.csv"):
        self.data_dir = data_dir
        self.dataset, self.index_to_asset_map = self._preprocess_dataset()

    def _read_dataset_as_df(self):
        return pd.read_csv(filepath_or_buffer=self.data_dir, header=0)

    def _preprocess_dataset(self):
        df = self._read_dataset_as_df()
        grouped_by_asset_class = df.groupby('Name')
        mapping = dict()
        output = list()

        index = 1
        for name, group in grouped_by_asset_class:
            group_values = group[['open', 'high', 'low', 'close']].values
            ## Hard coded as 1259 as preliminary investigation 
            ## was done and we include only stock whose prices
            ## are tracked over the same time period of 1259.
            if len(group_values) != 1259:
                continue
            group_tensor = torch.tensor(group_values)
            output.append(group_tensor)
            mapping[name] = index
            index = index + 1

        return torch.stack(output), mapping
    
    def get_assets_at_random(self, n):
        total_num_asset = self.dataset.shape[0]
        rand_indices = [random.randint(0, total_num_asset) for _ in range(n)]
        return self.dataset[rand_indices, :, :]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

## We wouldn't need to load it into a dataloader
# train_dataloader = DataLoader(StockDataset(), batch_size=128, shuffle=True)


