# Convolutional Autoencoder

This is a stand-alone program that trains a convolutional autoencoder on pre-processed and post-processed MRI scans. The scans are from the ADNI dataset, and the post-processed scans are processed with FreeSurfer. The goal of the autoencoder is to approximate the post-processing done with FreeSurfer, which eliminates unnecessary elements from the MRI, such as the neck, the skull, and other undesirable features.

## Notes

1. At least 16 images per batch can be run on 1 GTX1080 TI GPU,

## TODO

1. Investigate why loss becomes NaN after a few iterations,
