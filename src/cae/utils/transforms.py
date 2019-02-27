import torch.nn.functional as F

class OrientFSImage(object):
    '''
    An object used by Dataset class to orient the FreeSurfer MRI images so they match those of the preprocessed images.
    '''
    def __init__(self):
        pass

    def __call__(self, image):
        # Flipping the coronal (dim=1) and axial (dim=2) plane.
        return image.flip(1).flip(2)

class PadPreprocImage(object):
    '''
    Pads the image on both sides so it becomes a cube of size 256x256x256.
    '''
    def __init__(self):
        pass

    def __call__(self, image):
        dim = tuple(image.shape)
        pad_amount = [0, 0, 0, 0, 0, 0]

        for idx in range(len(dim)):
            if dim[idx] < 256:
                padding = (256 - dim[idx]) // 2
                pad_amount[idx*2] = padding
                pad_amount[idx*2+1] = padding

        pad_params = { "mode": "constant", "value": 0 }
        padded_image = F.pad(image,
                             tuple(reversed(pad_amount)),
                             **pad_params)

        return padded_image

class RangeNormalization(object):
    '''
    Normalize the pixel values to between 0 and 1.
    '''
    def __init__(self):
        pass

    def __call__(self, image):
        return image / image.max()

class MeanStdNormalization(object):
    '''
    Normalize the pixel values to between -1 and 1 by subtracting the mean and dividing by the standard deviation.
    '''
    def __init__(self):
        pass

    def __call__(self, image):
        return (image - image.mean()).std()
