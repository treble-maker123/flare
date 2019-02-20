# This preprocessing script produces a pandas DataFrame of file-path mapping
# between pre-processed MRIs and FreeSurfer post-processed MRIs.

# This script should be run from /disease_forecasting/src/cae/utils/

import os
import pickle
import pandas as pd
import numpy as np
import nibabel as nib

from fnmatch import fnmatch
from pdb import set_trace

class FileMapping:
    # Images are stored in /mnt/nfs/work1/mfiterau/ADNI_data/
    DATA_PATH = "/mnt/nfs/work1/mfiterau/ADNI_data/"
    PREPROC_PATHS = [
        DATA_PATH + "data/1",
        DATA_PATH + "data/2",
        DATA_PATH + "data/3",
        DATA_PATH + "data/4",
        DATA_PATH + "data/5",
        DATA_PATH + "data/6",
        DATA_PATH + "data/7",
        DATA_PATH + "data/8",
        DATA_PATH + "data/9",
        DATA_PATH + "data/10"
    ]
    POSTPROC_PATHS = [
        DATA_PATH + "surfer/ADNI"
    ]

    EXTENSION = "*.nii"

    # Example values for the columns,
    #
    # subject_id: 002_S_0413
    # date: 2008-07-31_14_39_56.0/S54591
    # image_id: S54591
    # preproc_path: /mnt/nfs/work1/mfiterau/ADNI_data/data/1/ADNI/002_S_0413/MPR__GradWarp__B1_Correction__N3__Scaled/2008-07-31_14_39_56.0/S54591/ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20081015122825655_S54591_I120917.nii
    # postproc_path: /mnt/nfs/work1/mfiterau/ADNI_data/surfer/ADNI/002_S_0413/FreeSurfer_Cross-Sectional_Processing_brainmask/2008-07-31_14_39_56.0/S54591/ADNI_002_S_0413_VOLUME_FreeSurfer_Cross-Sectional_Processing_Br_20100615150324572.brainmask.nii
    COLUMNS = ["subject_id", "date", "image_id","preproc_path", "postproc_path"]

    def __init__(self, **kwargs):
        '''
        Args:
            **kwargs:
        '''
        self.mapping = pd.DataFrame(columns=self.COLUMNS)

    def generate_files_manifest(self, **kwargs):
        '''
        Generates a pickle file containing a pandas.DataFrame object that maps pre-processed image paths to its corresponding post-procesed image paths.

        Args:
            **kwargs:
                output_name (string): Name of the output file, defaults to "files_manifest.pickle"
                output_path (string): Directory to output the file, defaults to "outputs/"
                verbose (bool): Whether to log additional data, defaults to True
        '''
        output_name = kwargs.get("output_name", "files_manifest.pickle")
        output_path = kwargs.get("output_path", "../outputs/")
        verbose = kwargs.get("verbose", True)

        if os.path.exists(output_path + output_name):
            print("A file with the name {} already exists in {}, exiting.".
                    format(output_name, output_path))
            return

        for path in self.PREPROC_PATHS:
            if verbose: print("Started processing files under {}".format(path))

            mappings = self._get_paths(path, category="preproc")

            if verbose:
                print("\t{} files processed under {}"
                        .format(len(mappings), path))

        print("Started with post-processed files.")
        mappings = self._get_paths(self.POSTPROC_PATHS[0], category="postproc")
        print("\t{} post-processed files processed.".format(len(mappings)))

        with open(output_path + output_name, 'wb') as file:
            pickle.dump(self.mapping, file)

    def _get_paths(self, root, **kwargs):
        '''
        Goes through all of the files under the root directory, identify the ones with .nii extension, and builds self.mapping accordingly.

        This method modifies self.mapping directly.

        Args:
            **kwargs:
                category (string): Either "preproc" or "postproc", defaults to "preproc"
        '''
        mapping = []

        category = kwargs.get("category", "preproc")
        assert category in ["preproc", "postproc"], \
                "Unrecognized category passed to _get_paths."

        for path, subdirs, files in os.walk(root):
            for file_name in files:
                # skip files not ending in ".nii"
                if not fnmatch(file_name, self.EXTENSION):
                    continue

                # get metadata from path
                parts = path.split("/")
                parts_len = 13 if category == "preproc" else 12

                if len(parts) != parts_len:
                    print("Irregular file path, skipped... {}" \
                            .format(file_name))
                    continue

                file_path = "{}/{}".format(path, file_name)

                # check if the image is valid, skip the ones that are not.
                try:
                    image = nib.load(file_path) \
                               .get_fdata() \
                               .squeeze()

                    if np.isnan(image).sum() > 0:
                        raise Exception("Image corrupted.")
                except Exception as e:
                    print("File corrupted, skipping.")
                    continue

                if category == "preproc":
                    subject_id = parts[9]
                    date = parts[11]
                    image_id = parts[12]
                    entry = [subject_id, date, image_id, file_path, None]
                else:
                    subject_id = parts[8]
                    date = parts[10]
                    image_id = parts[11]
                    entry = [subject_id, date, image_id, None, file_path]

                    m = self.mapping
                    idx = (m["subject_id"] == subject_id) & \
                          (m["date"] ==  date) & \
                          (m["image_id"] == image_id)

                    # skip the ones with 0 or more than 1 matches
                    if len(m[idx].index) == 1:
                        self.mapping.loc[idx, "postproc_path"] = file_path

                mapping.append(entry)

        if category == "preproc":
            mapping_df = pd.DataFrame(mapping, columns=self.COLUMNS)
            self.mapping = self.mapping.append(mapping_df, ignore_index=True)

        return mapping

if __name__ == "__main__":
    print("----- Started mapping.py -----")
    mapping = FileMapping()
    mapping.generate_files_manifest(verbose=True)
    print("----- Finished mapping.py -----")
