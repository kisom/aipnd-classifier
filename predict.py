#!/usr/bin/env python3

import numpy as np
import PIL.Image as Image

# Image Preprocessing

# You'll want to use PIL to load the image (documentation). It's best to write a
# function that preprocesses the image so it can be used as input for the model.
# This function should process the images in the same manner used for training.

# First, resize the images where the shortest side is 256 pixels, keeping the
# aspect ratio. This can be done with the thumbnail or resize methods. Then
# you'll need to crop out the center 224x224 portion of the image.

# Color channels of images are typically encoded as integers 0-255, but the
# model expected floats 0-1. You'll need to convert the values. It's easiest
# with a Numpy array, which you can get from a PIL image like so np_image =
# np.array(pil_image).

# As before, the network expects the images to be normalized in a specific way.
# For the means, it's [0.485, 0.456, 0.406] and for the standard deviations
# [0.229, 0.224, 0.225]. You'll want to subtract the means from each color
# channel, then divide by the standard deviation.

# And finally, PyTorch expects the color channel to be the first dimension but
# it's the third dimension in the PIL image and Numpy array. You can reorder
# dimensions using ndarray.transpose. The color channel needs to be first and
# retain the order of the other two dimensions.FB
def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    image.thumbnail((256, 256))
    width, height = image.size
    cropx = min(width, 224)
    cropy = min(height, 224)

    # note: PIL seems to handle float sizes okay.
    box = ((width/2)-(cropx/2), ((height/2)-(cropy/2)), 
           (width/2)+(cropx/2), ((height/2)+(cropy/2)))
    image = image.crop(box)

    np_image = np.array(image)
    (x, y, z) = np_image.shape
    np_image = np_image.reshape(x * y, z)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = np_image - mean 
    np_image = np_image / std
    np_image = np_image.reshape(x, y, z)

    return np_image

def test_process_image(image_path='test_image.jpg'):
    return process_image(Image.open(image_path))

# To check your work, the function below converts a PyTorch tensor and displays
# it in the notebook. If your `process_image` function works, running the output
# through this function should return the original image (except for the cropped
# out portions).
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


# Class Prediction

# Once you can get images in the correct format, it's time to write a function
# for making predictions with your model. A common practice is to predict the
# top 5 or so (usually called top-𝐾) most probable classes. You'll want to
# calculate the class probabilities then find the 𝐾 largest values.

# To get the top 𝐾 largest values in a tensor use x.topk(k). This method
# returns both the highest k probabilities and the indices of those
# probabilities corresponding to the classes. You need to convert from these
# indices to the actual class labels using class_to_idx which hopefully you
# added to the model or from an ImageFolder you used to load the data (see
# here). Make sure to invert the dictionary so you get a mapping from index to
# class as well.

# Again, this method should take a path to an image and a model checkpoint, then
# return the probabilities and classes.

# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
def predict(image_path, model, topk=5):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """

    # TODO: Implement the code to predict the class from an image file
    pass
