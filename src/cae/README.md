# Convolutional Autoencoder

This is a stand-alone program that trains a convolutional autoencoder on pre-processed and post-processed MRI scans. The scans are from the ADNI dataset, and the post-processed scans are processed with FreeSurfer. The goal of the autoencoder is to approximate the post-processing done with FreeSurfer, which eliminates unnecessary elements from the MRI, such as the neck, the skull, and other undesirable features.

## Notes

1. At least 16 images per batch can be run on 1 GTX1080 TI GPU,

### Current Setup on Gypsum

All numbers depend on the number of GPUs available, therefore adjust accordingly.

1. 8 GPUs,

2. 16 images per GPU = 128 images,

3. 4 DataLoader workers per 1 GPU = 32 workers,

4. 32 workers + 2 extra = 34 CPUs,

5. 1GB per 1 CPU = 32GB (mem=32000)
