from torch.utils.data import Dataset
import h5py
import numpy as np
import json
from data.utils import subset_dict_by_filename, filter_labels_by_threshold
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
        self.path = path
        self.transform = transform
        self.partition = partition
        self.modalities = modalities
        self.mono_strict = mono_strict
        data_path = path + split + "_filenames.lst"
        with open(data_path, "r") as file:
            self.data_list = [line.strip() for line in file.readlines()]
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
            with rasterio.open(self.path + "aerial/" + name) as f:
                output["aerial"] = torch.FloatTensor(f.read())

        # with h5py.File(
        #     self.path + "sentinel/" + ".".join(name.split(".")[:-1]) + ".h5", "r"
        # ) as file:
        #     if "s2" in self.modalities:
        #         output["s2"] = torch.tensor(file["sen-2-data"][:])
        #         output["s2_dates"] = day_number_in_year(
        #             file["sen-2-products"][:], place=2
        #         )
        #         N = len(output["s2_dates"])
        #         if N > 50:
        #             random_indices = torch.randperm(N)[:50]
        #             output["s2"] = output["s2"][random_indices]
        #             output["s2_dates"] = output["s2_dates"][random_indices]

        if "s2-mono" in self.modalities:
            with rasterio.open(self.path + "s2/60m/" + name) as f:
                numpy_array = f.read()
            numpy_array = numpy_array.astype(np.float32)
            output["s2-mono"] = torch.FloatTensor(numpy_array)
            if self.mono_strict:
                output["s2-mono"] = output["s2-mono"][:10, :, :]

        if "s2-4season-median" in self.modalities:
            with h5py.File(
                self.path + "sentinel/" + ".".join(name.split(".")[:-1]) + ".h5", "r"
            ) as file:
                output_inter = torch.tensor(file["sen-2-data"][:])
                dates = day_number_in_year(file["sen-2-products"][:], place=2)
            l = []
            for i in range(4):
                mask = (dates >= 92 * i) & (dates < 92 * (i + 1))
                if sum(mask) > 0:
                    r, _ = torch.median(output_inter[mask], dim=0)
                    l.append(r)
                else:
                    l.append(
                        torch.zeros(
                            (
                                output_inter.shape[1],
                                output_inter.shape[-2],
                                output_inter.shape[-1],
                            )
                        )
                    )
            output["s2-4season-median"] = torch.cat(l)

        if "s2-median" in self.modalities:
            with h5py.File(
                self.path + "sentinel/" + ".".join(name.split(".")[:-1]) + ".h5", "r"
            ) as file:
                output["s2-median"], _ = torch.median(
                    torch.tensor(file["sen-2-data"][:]), dim=0
                )

        return self.transform(output)

    def __len__(self):
        return len(self.data_list)
