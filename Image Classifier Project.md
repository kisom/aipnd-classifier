
# Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

<img src='assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.


```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
```

## Load the data

Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.

The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
 


```python
import dataset

flowerset = dataset.Dataset("flowers", 32)
```

### Label mapping

You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.


```python
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
```

# Building and training the classifier

Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.

We're going to leave this part up to you. If you want to talk through it with someone, chat with your fellow students! You can also ask questions on the forums or join the instructors in office hours.

Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:

* Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
* Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
* Train the classifier layers using backpropagation using the pre-trained network to get the features
* Track the loss and accuracy on the validation set to determine the best hyperparameters

We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!

When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.


```python
import gym, model
```


```python
hyper_params = {
    'architecture': 'vgg16',
    'criterion': 'NLLLoss',
    'dropout': 0.5,
    'epochs': 6,
    'layers': [4096, 4096],
    'learning_rate': 0.001,
    'nfeatures': 102,
    'ninputs': 32,
    'optimizer': 'Adam',
}
flower_model = model.Model(hyper_params, cat_to_name, flowerset.class_to_idx)
flower_model.train()
```

    INFO: network is in training mode



```python
flower_gym = gym.Gym(flower_model, flowerset, 32)
```

    INFO: GPU available



```python
flower_gym.train()
```

    INFO: starting deep learning via cuda
    INFO: network is in training mode
    INFO: starting epoch: 0
    INFO: 0:00:47.876054     32|epoch: 1/6; loss: 4.9425
    INFO: 0:00:47.757795     64|epoch: 1/6; loss: 3.2358
    INFO: 0:00:48.176509     96|epoch: 1/6; loss: 2.6345
    INFO: running validation evaluation
    INFO: network is in evaluation mode
    INFO: 0:00:21.156059: validation accuracy over 818 test images: 57.95% (474/818)
    INFO: epoch completed in: 0:02:54.906632
    ------------------------------------------------------------------------
    INFO: network is in training mode
    INFO: starting epoch: 1
    INFO: 0:00:37.754776    128|epoch: 2/6; loss: 1.7923
    INFO: 0:00:47.967833    160|epoch: 2/6; loss: 2.0898
    INFO: 0:00:48.028213    192|epoch: 2/6; loss: 2.0815
    INFO: running validation evaluation
    INFO: network is in evaluation mode
    INFO: 0:00:21.227740: validation accuracy over 818 test images: 68.7% (562/818)
    INFO: epoch completed in: 0:02:55.164295
    ------------------------------------------------------------------------
    INFO: network is in training mode
    INFO: starting epoch: 2
    INFO: 0:00:27.005811    224|epoch: 3/6; loss: 1.0624
    INFO: 0:00:48.141598    256|epoch: 3/6; loss: 1.8154
    INFO: 0:00:48.278903    288|epoch: 3/6; loss: 1.9138
    INFO: running validation evaluation
    INFO: network is in evaluation mode
    INFO: 0:00:21.285769: validation accuracy over 818 test images: 71.27% (583/818)
    INFO: epoch completed in: 0:02:55.577171
    ------------------------------------------------------------------------
    INFO: network is in training mode
    INFO: starting epoch: 3
    INFO: 0:00:16.505750    320|epoch: 4/6; loss: 0.6268
    INFO: 0:00:48.327448    352|epoch: 4/6; loss: 1.7338
    INFO: 0:00:48.169804    384|epoch: 4/6; loss: 1.6873
    INFO: running validation evaluation
    INFO: network is in evaluation mode
    INFO: 0:00:21.333538: validation accuracy over 818 test images: 77.26% (632/818)
    INFO: epoch completed in: 0:02:55.724296
    ------------------------------------------------------------------------
    INFO: network is in training mode
    INFO: starting epoch: 4
    INFO: 0:00:05.998830    416|epoch: 5/6; loss: 0.2014
    INFO: 0:00:47.987793    448|epoch: 5/6; loss: 1.5568
    INFO: 0:00:48.174805    480|epoch: 5/6; loss: 1.5112
    INFO: 0:00:48.197146    512|epoch: 5/6; loss: 1.6086
    INFO: running validation evaluation
    INFO: network is in evaluation mode
    INFO: 0:00:21.236486: validation accuracy over 818 test images: 75.92% (621/818)
    WARN: accuracy has decreased
    INFO: epoch completed in: 0:02:55.341113
    ------------------------------------------------------------------------
    INFO: network is in training mode
    INFO: starting epoch: 5
    INFO: 0:00:43.639985    544|epoch: 6/6; loss: 1.4140
    INFO: 0:00:48.325554    576|epoch: 6/6; loss: 1.4664
    INFO: 0:00:48.282197    608|epoch: 6/6; loss: 1.5643
    INFO: running validation evaluation
    INFO: network is in evaluation mode
    INFO: 0:00:21.242175: validation accuracy over 818 test images: 79.83% (653/818)
    INFO: epoch completed in: 0:02:55.797475
    ------------------------------------------------------------------------
    INFO: training completed in 0:17:32.613345


## Testing your network

It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.


```python
flower_gym.evaluate()
```

    INFO: running testing evaluation
    INFO: network is in evaluation mode
    INFO: 0:00:21.116342: testing accuracy over 819 test images: 79.0% (647/819)





    0.78998778998779



## Save the checkpoint

Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.

```model.class_to_idx = image_datasets['train'].class_to_idx```

Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.


```python
flower_model.save('checkpoint.dat')
```

## Loading the checkpoint

At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.


```python
restored_model = model.load('checkpoint.dat')
```

# Inference for classification

Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```

First you'll need to handle processing the input image such that it can be used in your network. 

## Image Preprocessing

You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 

First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.

As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 

And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.


```python
def process_image(path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """
    return pipeline(Image.open(path))

def pipeline(image):
    return transforms.Compose([
    transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(image)
```

To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).


```python
from PIL import Image

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

_, ax = plt.subplots()
image = Image.open('test_image.jpg')
ax.imshow(np.array(image))
imshow(process_image('test_image.jpg'))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f562b46aba8>




![png](output_20_1.png)



![png](output_20_2.png)


## Class Prediction

Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.

To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```


```python
def predict(image_path, model, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """

    model.eval()
    image = process_image(image_path)
    # pylint: disable=E1101
    probs, indices = torch.exp(model.forward(image.unsqueeze(0))).topk(topk)
    # pylint: enable=E1101
    probs.squeeze_(0)
    indices.squeeze_(0)
    classes = [model.idx_to_class[i.item()] for i in indices]
    probs = [prob.item() for prob in probs]
    return probs, classes

```

## Sanity Checking

Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:

<img src='assets/inference_example.png' width=300px>

You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.


```python
probs, classes = predict('test_image.jpg', restored_model, 5)
labels = [cat_to_name[cls] for cls in classes]
```

    INFO: network is in evaluation mode



```python
figure = plt.figure(figsize=[5,5])

# set up a subplot with one column and two rows.
image_sub = plt.subplot(2, 1, 1)

# show no labels but do add a title, then show the image.
image_sub.tick_params(left=None, labelleft=None, bottom=None, labelbottom=None)
image_sub.set_title(labels[0])
imshow(process_image('test_image.jpg'), ax=image_sub)

pred_sub = plt.subplot(2, 1, 2)
yticks = np.arange(5)
plt.barh(yticks, probs[::-1], align='center')
plt.yticks(yticks, labels[::-1])
plt.xticks(rotation=90)
plt.show()
```


![png](output_25_0.png)



```python

```
