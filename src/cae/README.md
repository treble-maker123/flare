# Convolutional Autoencoder

This is a stand-alone program that trains a convolutional autoencoder on pre-processed and post-processed MRI scans. The scans are from the ADNI dataset, and the post-processed scans are processed with FreeSurfer. The goal of the autoencoder is to approximate the post-processing done with FreeSurfer, which eliminates unnecessary elements from the MRI, such as the neck, the skull, and other undesirable features.

## Notes

1. At least 16 images per batch can be run on 1 GTX1080 TI GPU with the Vanilla model,

2. A mapping file is needed to run. The mapping file contains path mapping between preprocess and postprocessed images, and can be generated with the script in `utils/mapping.py`. Be warned that running this script will take a look time, as it also opens each file and checks for error, so that corrupted files are excluded, and it does so single-threaded. Alternatively, you can find the `mapping_manifest.pickle` in `/mnt/nfs/work1/mfiterau/zguan/work1/disease_forecasting/src/cae/outputs/`.

### Current Setup on Gypsum

All numbers depend on the number of GPUs available, therefore adjust accordingly.

1. 2 GPUs,

2. 16 images per GPU = 32 images,

3. 4 DataLoader workers per 1 GPU = 8 workers,

4. 8 workers + 2 extra = 10 CPUs,

5. 45GB per 1 GPU = 90GB (mem=90000)
