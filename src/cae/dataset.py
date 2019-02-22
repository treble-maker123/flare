import os
import pickle
import numpy as np
import nibabel as nib
import multiprocessing as mp
import torchvision
import torchvision.transforms as T

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.loader import invalid_collate
from utils.transforms import OrientFSImage, PadPreprocImage

from pdb import set_trace

class ADNIAutoEncDataset(Dataset):
    '''
    ADNI dataset for training auto-encoder. This dataset relies on a mapping file, which is generated with the mapping.py script in utils/. The mapping.py script makes certain assumptions about the location of all of the data files.

    Args:
        **kwargs:
            transforms (List): A list of torchvision.transforms functions, see https://pytorch.org/docs/stable/torchvision/transforms.html for more information.

            mode (string): "train" for training split, "valid" for validation split, and "all" for the whole set, defaults to "all".

            valid_split (float): Percentage of data reserved for validation, ignored if mode is set to "train" or "all, defaults to 0.2.

            limit (int): Size limit for the dataset, used for debugging. This is the total number of images to include for both the training set and validation set. For example, limit=10 with a valid_split of 0.2 will assign 8 images to the training set, 2 images to the validation set. Setting to -1 means to include the whole set. Defaults to -1.
    '''
    def __init__(self, **kwargs):
        mapping_path = kwargs.get("mapping_path",
                                  "outputs/files_manifest.pickle")
        preproc_transforms = kwargs.get("preproc_transforms",
                                        [ T.ToTensor(),
                                          PadPreprocImage() ])
        postproc_transforms = kwargs.get("postproc_transforms",
                                         [ T.ToTensor(),
                                           OrientFSImage() ])
        mode = kwargs.get("mode", "all")
        valid_split = kwargs.get("valid_split", 0.2)
        limit = kwargs.get("limit", -1)

        if not os.path.exists(mapping_path):
            raise Exception("Failed to create dataset, \"{}\" does not exist! Run \"utils/mapping.py\" script to generate mapping."
                .format(mapping_path))

        if mode not in ["train", "valid", "all"]:
            raise Exception("Invalid mode: {}".format(mode))

        if not 0.0 < valid_split < 1.0:
            raise Exception("Invalid validation split percentage: {}"
                                .format(valid_split))

        with open(mapping_path, "rb") as file:
            self.df = pickle.load(file)

        self.preproc_transforms = T.Compose(preproc_transforms)
        self.postproc_transforms = T.Compose(postproc_transforms)

        self.df = self.df[self.df["postproc_path"].notnull()].reset_index()

        if limit != -1:
            assert limit > 0, "Invalid limit size: {}. Must be -1 or greater than 0".format(limit)
            self.df = self.df.iloc[:limit]
        else:
            print("Limiting total dataset size to: {}".format(limit))

        # train/validation split
        train_split = 1 - valid_split
        num_train = int(len(self.df.index) * train_split)
        if mode == "train":
            self.df = self.df.iloc[:num_train].reset_index()
        elif mode == "valid":
            self.df = self.df.iloc[num_train:].reset_index()

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        preproc_path, postproc_path = self._get_paths(idx)

        try:
            preproc_img = nib.load(preproc_path) \
                             .get_fdata() \
                             .squeeze()
            postproc_img = nib.load(postproc_path) \
                              .get_fdata() \
                              .squeeze()

            if (np.isnan(preproc_img).sum() > 0) or \
               (np.isnan(postproc_img).sum() > 0):
                raise Exception("Corrupted image {}.".format(idx))
        except Exception as e:
            print("Failed to load #{}, skipping.".format(idx))
            return None, None

        preproc_img = self.preproc_transforms(preproc_img)
        postproc_img = self.postproc_transforms(postproc_img)

        # Add a "channel" dimension
        preproc_img = preproc_img.unsqueeze(0)
        postproc_img = postproc_img.unsqueeze(0)
        return preproc_img, postproc_img

    def process_images(self, fn, **kwargs):
        '''
        Takes a function and performs it on all images. This method uses DataLoader to parallelize the operation.

        Args:
            fn (function): A function to be applied to each individual images and has a signature of fn(preproc, postproc).
            **kwargs:
                max_proc (int): The maximum number of processes to spawn for DataLoader workers, defaults to half the number of cpu cores.

        Returns:
            dict: A dictionary of outputs from the fn.
        '''
        max_proc = kwargs.get("max_proc", mp.cpu_count() // 2)
        num_workers = max(max_proc, 1)
        loader = DataLoader(self,
                            batch_size=1,
                            num_workers=max_proc,
                            collate_fn=invalid_collate)
        outputs = []

        print("Running process_images with {} DataLoader workers"
                .format(num_workers))
        for idx, images in enumerate(loader):
            if len(images) == 0:
                continue
            preproc, postproc = images

            for i in range(len(preproc)):
                if preproc[i] is None or postproc[i] is None:
                    continue

                output = fn(preproc[i], postproc[i])

                if output is not None:
                    outputs.append(output)
        print("Done!")

        return outputs

    def _get_paths(self, idx):
        '''
        Returns the file paths for the given index.

        Args:
            idx (int): Index of the paths
        Returns:
            tuple: A pair of strings containing the preprocess and post-processed image paths.
        '''
        preproc_path = self.df.preproc_path.iloc[idx]
        postproc_path = self.df.postproc_path.iloc[idx]

        return preproc_path, postproc_path

    def _get_dims(self, preproc, postproc):
        '''
        A function for process_images, returns the shapes of preproc and postproc images.

        Args:
            preproc (torch.Tensor): Preprocessed image
            postproc (torch.Tensor): Postprocessed image

        Returns:
            tuple: Two tuples of shapes for preproc and postproc
        '''
        return (tuple(preproc.shape), tuple(postproc.shape))


class ADNIClassDataset(Dataset):
    '''
    ADNI dataset for training auto-encoder. This dataset relies on a mapping file, which is generated with the mapping.py script in utils/. The mapping.py script makes certain assumptions about the location of all of the data files.

    Args:
        **kwargs:
            transforms (List): A list of torchvision.transforms functions, see https://pytorch.org/docs/stable/torchvision/transforms.html for more information.
            mode (string): "train" for training split, "valid" for validation split, and "all" for the whole set, defaults to "all".
            valid_split (float): Percentage of data reserved for validation, ignored if mode is set to "train" or "all, defaults to 0.2.
    '''
    def __init__(self, **kwargs):
        mapping_path = kwargs.get("mapping_path",
                                  "outputs/files_manifest.pickle")
        postproc_transforms = kwargs.get("postproc_transforms",
                                         [ T.ToTensor(),
                                           OrientFSImage() ])
        mode = kwargs.get("mode", "all")
        valid_split = kwargs.get("valid_split", 0.2)

        if not os.path.exists(mapping_path):
            raise Exception("Failed to create dataset, \"{}\" does not exist! Run \"utils/mapping.py\" script to generate mapping."
                .format(mapping_path))

        if mode not in ["train", "valid", "all"]:
            raise Exception("Invalid mode: {}".format(mode))

        if not 0.0 < valid_split < 1.0:
            raise Exception("Invalid validation split percentage: {}"
                                .format(valid_split))

        with open(mapping_path, "rb") as file:
            self.df = pickle.load(file)

        self.postproc_transforms = T.Compose(postproc_transforms)

        self.df = self.df[self.df["postproc_path"].notnull()].reset_index()

        self.df = self.df.iloc[:512]

        # train/validation split
        train_split = 1 - valid_split
        num_train = int(len(self.df.index) * train_split)
        if mode == "train":
            self.df = self.df.iloc[:num_train].reset_index()
        elif mode == "valid":
            self.df = self.df.iloc[num_train:].reset_index()

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        postproc_path = self._get_paths(idx)

        try:
            postproc_img = nib.load(postproc_path) \
                              .get_fdata() \
                              .squeeze()

            if (np.isnan(postproc_img).sum() > 0):
                raise Exception("Corrupted image {}.".format(idx))
        except Exception as e:
            print("Failed to load #{}, skipping.".format(idx))
            return None, None

        postproc_img = self.postproc_transforms(postproc_img)

        # Add a "channel" dimension
        postproc_img = postproc_img.unsqueeze(0)

        # Find the label for corresponding patient data
        subject_id = self._get_subjectID(idx)
        features = np.loadtxt("../../data/features2.csv", delimiter=',', skiprows=1, dtype='str')
        label = features[np.where(features[:,0] == subject_id), 1]
        label = str(label)
        output = 0
        if label == "AD":
            output = 2
        elif label == "MCI":
            output = 1

        return postproc_img, output

    def process_images(self, fn, **kwargs):
        '''
        Takes a function and performs it on all images. This method uses DataLoader to parallelize the operation.

        Args:
            fn (function): A function to be applied to each individual images and has a signature of fn(preproc, postproc).
            **kwargs:
                max_proc (int): The maximum number of processes to spawn for DataLoader workers, defaults to half the number of cpu cores.

        Returns:
            dict: A dictionary of outputs from the fn.
        '''
        max_proc = kwargs.get("max_proc", mp.cpu_count() // 2)
        num_workers = max(max_proc, 1)
        #batchsize = kwargs.get("batch_size", 16)
        loader = DataLoader(self,
                            batch_size=16,
                            num_workers=max_proc,
                            collate_fn=invalid_collate)
        outputs = []

        print("Running process_images with {} DataLoader workers"
                .format(num_workers))
        for idx, images in enumerate(loader):
            if len(images) == 0:
                continue
            postproc = images

            for i in range(len(postproc)):
                if postproc[i] is None:
                    continue

                output = fn(postproc[i])

                if output is not None:
                    outputs.append(output)
        print("Done!")

        return outputs

    def _get_paths(self, idx):
        '''
        Returns the file paths for the given index.

        Args:
            idx (int): Index of the paths
        Returns:
            tuple: A pair of strings containing the preprocess and post-processed image paths.
        '''
        postproc_path = self.df.postproc_path.iloc[idx]

        return postproc_path

    def _get_subjectID(self, idx):
        '''
        Returns the patient subject ID for the given index.

        Args:
            idx (int): Index of the paths
        Returns:
            tuple: A pair of strings containing the preprocess and post-processed image paths.
        '''
        subject_id = self.df.subject_id.iloc[idx]

        return subject_id

    def _get_dims(self, postproc):
        '''
        A function for process_images, returns the shapes of preproc and postproc images.

        Args:
            preproc (torch.Tensor): Preprocessed image
            postproc (torch.Tensor): Postprocessed image

        Returns:
            tuple: Two tuples of shapes for preproc and postproc
        '''
        return (tuple(postproc.shape))


if __name__ == "__main__":
    dataset = ADNIAutoEncDataset()

    # Get the unique shapes for preprocessed and post-processed images
    # shapes = dataset.process_images(dataset._get_dims)
    # preproc_shapes = set(map(lambda x: x[0], shapes))
    # postproc_shapes = set(map(lambda x: x[1], shapes))
    # preproc_shapes -> [(256, 170, 256), (146, 256, 256), (124, 256, 256), (166, 256, 256), (160, 192, 192), (192, 160, 192), (170, 256, 256), (180, 256, 256), (184, 256, 256), (162, 256, 256)]
    # (166, 256, 256) = 694
    # (180, 256, 256) = 264
    # (160, 192, 192) = 621
    # (170, 256, 256) = 46
    # (184, 256, 256) = 9
    # (256, 170, 256) = 1
    # (146, 256, 256) = 1
    # (192, 160, 192) = 1
    # (124, 256, 256) = 1
    # (162, 256, 256) = 2
    # postproc_shapes -> [(256, 256, 256)]
