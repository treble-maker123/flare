import os
import pickle
import pandas as pd
import numpy as np
import nibabel as nib
import torchvision.transforms as T

from pdb import set_trace
from torch.utils.data import Dataset
from utils.transforms import RangeNormalization, NaNToNum, PadToSameDim3D
from sklearn.preprocessing import LabelEncoder

class NormalizedDataset(Dataset):
    '''CLINICA-normalized dataset for classification task.
    '''
    VALID_MODES = ["train", "valid", "test", "all"]
    VALID_TASKS = ["pretrain", "classify"]

    def __init__(self, **kwargs):
        self.config = kwargs.get("config", {
            "image_col": "misc",
            "label_col": "label",
            "label_path": "outputs/normalized_mapping.pickle"
        })
        # limit for the size of the dataset, for debugging purposes
        self.limit = kwargs.get("limit", -1)
        self.verbose = kwargs.get("verbose", self.config["verbose"])

        transforms = kwargs.get("transforms", [
            T.ToTensor(),
            PadToSameDim3D(),
            NaNToNum(),
            RangeNormalization()
        ])
        self.transforms = T.Compose(transforms)

        # name of the image column in the dataframe
        self.image_col = self.config["image_col"]
        # name of the label column in the dataframe
        self.label_col = self.config["label_col"]

        mapping_path = kwargs.get("mapping_path",
                                  self.config["label_path"])
        mode = kwargs.get("mode", "all")
        task = kwargs.get("task", "classify")
        valid_split = kwargs.get("valid_split", 0.2)
        test_split = kwargs.get("test_split", 0.0)

        df, self.label_encoder = self._get_data(mapping_path)
        self.dataframe = self._split_data(df, valid_split, test_split, mode,
                                          task)

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, idx):
        image_path = self.dataframe[self.image_col].iloc[idx]
        label = self.dataframe[self.label_col].iloc[idx]
        encoded_label = self.label_encoder.transform([label])[0]

        try:
            image = nib.load(image_path) \
                       .get_fdata() \
                       .squeeze()
        except Exception as e:
            print("Failed to load #{}: {}".format(idx, image_path))
            return None, None

        # unsqueeze adds a "channel" dimension
        transformed_image = self.transforms(image).unsqueeze(0)

        if self.verbose:
            print("Fetched image (label: {}/{}) from {}."
                    .format(label, encoded_label, image_path))

        return transformed_image, encoded_label

    # ==============================
    # Helper Methods
    # ==============================
    def _split_data(self, df, valid_split, test_split, mode, task):
        if mode not in self.VALID_MODES:
            raise Exception("Invalid mode: {}. Valid options are {}"
                                .format(mode, self.VALID_MODES))

        if task not in self.VALID_TASKS:
            raise Exception("Invalid task: {}. Valid options are {}"
                                .format(task, self.VALID_TASKS))

        if not 0.0 <= valid_split <= 1.0:
            raise Exception("Invalid validation split percentage: {}"
                                .format(valid_split))

        if not 0.0 <= test_split <= 1.0:
            raise Exception("Invalid test split percentage: {}"
                                .format(test_split))

        if (valid_split + test_split) >= 1.0:
            raise Exception("valid_split + test_split ({}) is greater than or equal to 1.0".format(valid_split + test_split))

        ad = df[df[self.label_col] == "AD"]
        mci = df[df[self.label_col] == "MCI"]
        cn = df[df[self.label_col] == "CN"]
        size = min(len(ad.index), len(mci.index), len(cn.index)) \
                if self.limit == -1 else self.limit

        if task == "classify":
            ad = self._split_dataframe(ad[:size], valid_split, test_split, mode)
            mci = self._split_dataframe(mci[:size], valid_split, test_split,
                                        mode)
            cn = self._split_dataframe(cn[:size], valid_split, test_split, mode)

            print("Class distribution for {} {}: {} AD, {} MCI, {} CN"
                    .format(task, mode, len(ad.index), len(mci.index),
                            len(cn.index)))
        elif task == "pretrain":
            ad = self._split_dataframe(ad[size:], valid_split, test_split, mode)
            mci = self._split_dataframe(mci[size:], valid_split, test_split,
                                        mode)
            cn = self._split_dataframe(cn[size:], valid_split, test_split, mode)

            print("Class distribution for {} {}: {} AD, {} MCI, {} CN"
                    .format(task, mode, len(ad.index), len(mci.index),
                            len(cn.index)))

        return pd.concat([ad, mci, cn])

    def _split_dataframe(self, df, valid_split, test_split, mode):
        train_split = 1 - valid_split - test_split
        num_train = int(len(df.index) * train_split)
        num_valid = int(len(df.index) * valid_split)
        num_test = int(len(df.index) * test_split)

        if mode == "train":
            return df[:num_train].reset_index(drop=True)
        elif mode == "valid":
            start = num_train
            end = start + num_valid
            return df[start:end].reset_index(drop=True)
        elif mode == "test":
            start = num_train + num_valid
            end = start + num_test
            return df[start:end].reset_index(drop=True)
        else:
            return df[:]

    def _get_data(self, mapping_path):
        if not os.path.exists(mapping_path):
            raise Exception("Failed to create dataset, \"{}\" does not exist! Run \"utils/normalized_mapping.py\" script to generate mapping."
                .format(mapping_path))

        with open(mapping_path, "rb") as file:
            df = pickle.load(file)

        # filter out rows with empty label
        df = df[df[self.label_col].notnull()].reset_index()
        # filter out rows with empty image path
        df = df[df[self.image_col].notnull()].reset_index()

        # change LMCI and EMCI to MCI
        target = (df[self.label_col] == "LMCI") | \
                 (df[self.label_col] == "EMCI")
        df.loc[target, self.label_col] = "MCI"

        # setup labels encoder
        labels = df[self.label_col].unique()
        encoder = LabelEncoder()
        encoder.fit(labels)

        return df, encoder

if __name__ == "__main__":
    dataset = NormalizedDataset()
