class OrientFSImage(object):
    '''
    An object used by Dataset class to orient the FreeSurfer MRI images so they match those of the preprocessed images.
    '''
    def __init__(self):
        pass

    def __call__(self, image):
        # Flipping the coronal (dim=1) and axial (dim=2) plane.
        return image.flip(1).flip(2)
