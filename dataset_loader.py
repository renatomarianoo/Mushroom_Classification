from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.model_selection import train_test_split


class MushroomDataset(Dataset):
    """Define the dataset class"""

    def __init__(self, dataframe, transform=None):
        super().__init__()
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]  # path index=0
        label = self.dataframe.iloc[idx, 1]  # label index=1
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def dataset_separation_loader(
    df, transform, batch_size=16, val_size=0.3, test_size=0.35, state=42
):
    """
    Aims at splitting the data into traininig, validation and test sets.
    Proportions: 70%, 20%, 10%
    """

    train_df, temp_df = train_test_split(df, test_size=val_size, random_state=state)
    val_df, test_df = train_test_split(temp_df, test_size=test_size, random_state=state)

    train_dataset = MushroomDataset(dataframe=train_df, transform=transform)
    val_dataset = MushroomDataset(dataframe=val_df, transform=transform)
    test_dataset = MushroomDataset(dataframe=test_df, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )  # , num_workers=4 # no GPU :/
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
