from pathlib import Path

from torch.utils.data import Dataset
import h5py
import numpy as np
import json
from skmultilearn.model_selection import iterative_train_test_split
import torch
import rasterio
from datetime import datetime


def collate_fn(batch):
    """
    Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "label", "name"  and the other corresponding to the modalities used
    Returns:
        dict: dictionary with keys "label", "name"  and the other corresponding to the modalities used
    """
    keys = list(batch[0].keys())
    output = {}
    for key in ["s2", "s1-asc", "s1-des", "s1"]:
        if key in keys:
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack(
                [
                    torch.nn.functional.pad(
                        tensor, (0, 0, 0, 0, 0, 0, 0, max_size_0 - tensor.size(0))
                    )
                    for tensor in idx
                ],
                dim=0,
            )
            output[key] = stacked_tensor
            keys.remove(key)
            key = "_".join([key, "dates"])
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack(
                [
                    torch.nn.functional.pad(tensor, (0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ],
                dim=0,
            )
            output[key] = stacked_tensor
            keys.remove(key)
    if "name" in keys:
        output["name"] = [x["name"] for x in batch]
        keys.remove("name")
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output


def day_number_in_year(date_arr, place=4):
    day_number = []
    for date_string in date_arr:
        date_object = datetime.strptime(
            str(date_string).split("_")[place][:8], "%Y%m%d"
        )
        day_number.append(date_object.timetuple().tm_yday)  # Get the day of the year
    return torch.tensor(day_number)


def replace_nans_with_mean(batch_of_images):
    image_means = torch.nanmean(batch_of_images, dim=(3, 4), keepdim=True)
    image_means[torch.isnan(image_means)] = 0.0
    nan_mask = torch.isnan(batch_of_images)
    batch_of_images[nan_mask] = image_means.expand_as(batch_of_images)[nan_mask]
    return batch_of_images


class SocaDataset(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        classes: list = [],
        partition: float = 1.0,
        mono_strict: bool = False,
    ):
        """
        Initializes the dataset.
        Args:
            path (str): path to the dataset
            modalities (list): list of modalities to use
            transform (torch module): transform to apply to the data
            split (str): split to use (train, val, test)
            classes (list): name of the differrent classes
            partition (float): proportion of the dataset to keep
            mono_strict (bool): if True, puts monodate in same condition as multitemporal
        """
        self.path = Path(path)
        self.transform = transform
        self.partition = partition
        self.modalities = modalities
        self.mono_strict = mono_strict
        data_path = Path(self.path) / "s2_tiles_labeled"
        self.data_list = [line.name for line in data_path.glob("*")]
        self.collate_fn = collate_fn

    def __getitem__(self, i):
        """
        Returns an item from the dataset.
        Args:
            i (int): index of the item
        Returns:
            dict: dictionary with keys "label", "name" and the other corresponding to the modalities used
        """
        name = self.data_list[i]
        output = {"name": name}

        if "aerial" in self.modalities:
            with rasterio.open(self.path  / "drone_tiles_labeled" / name) as f:
                aerial = torch.FloatTensor(f.read())
                output["aerial"] = aerial

        with rasterio.open(self.path / "mask_tiles_labeled" / name) as f:
            label = torch.FloatTensor(f.read()) # in range 0 - 1
            # remove padded 255 vals:
            label[label == 255] = 0
            if "aerial" in self.modalities:
                # 1 where all pixels are white or blank
                invalid_mask = (aerial == 255).all(dim=0, keepdim=True).type(torch.int32) + (aerial == 0).all(dim=0, keepdim=True).type(torch.int32)
                # remove invalid from label, then subtract invalid to get: 0-no river, 1-river, -1 - invalid.
                masked_label = label * (1 - invalid_mask) - invalid_mask
            output["label"] = masked_label

        # B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12
        if "s2-mono" in self.modalities:
            with rasterio.open(self.path  / "s2_tiles_labeled" / name) as f:
                numpy_array = f.read()
            numpy_array = numpy_array.astype(np.float32)
            output["s2-mono"] = torch.FloatTensor(numpy_array)
            if self.mono_strict:
                output["s2-mono"] = output["s2-mono"][:10, :, :]

        return self.transform(output)

    def __len__(self):
        return len(self.data_list)
