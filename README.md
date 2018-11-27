# CNN preferred image (cnnpref)

Generating preferred image for the units in a CNN model.


## Version

version 1 (updated: 2018-11-27)


## Requirements

- Python 2.7
- Numpy 1.11.2
- Scipy 0.16.0
- Caffe with up-convolutional layer: https://github.com/dosovits/caffe-fr-chairs (Branch: deepsim) (both CPU and GPU installation are OK)


## Description

This repository contains Python codes for generating preferred image of the units in a CNN model.
Generating preferred image is based on the ‘activation maximum’ method [1,2] which synthesizes images such that the target units can have high activation (high values).
The loss function of this image generation problem is the values of the target units. 
The generation problem is solved by gradient based optimization algorithms.
The optimization starts with an initial image (random noise or black image), the derivative of the loss function with respect to the image is calculated by back-propagation via the CNN, then the image is updated with the derivative.
We use both gradient descent with momentum (we use the negative value of the target unit as the loss) and L-BFGS (https://en.wikipedia.org/wiki/Limited-memory_BFGS) as the optimization algorithm.

Inspired by the paper “Synthesizing the preferred inputs for neurons in neural networks via deep generator networks” [3], the optimization is also performed in the input layer of a deep generator network, in order to introduce image prior to the preferred image.
Here, the optimization starts with an initial features (random noise) of the input layer of the deep generator network, inputs the initial features to the generator to generate the initial image, which is further input to the CNN model.
The derivative in feature space of the CNN is back-propagated through the CNN to the image layer, and back-propagated further through the generator to the input layer of it, which is used to update the initial features of the input layer of the generator.

There are 4 variants of implementation of the preferred image generation algorithm:

- prefer_img_gd
- prefer_img_lbfgs
- prefer_img_dgn_gd
- prefer_img_dgn_lbfgs

The “prefer_img_gd” implements the preferred image generation algorithm using gradient descent (GD) with momentum as the optimization algorithm.

The “prefer_img_lbfgs” implements the preferred image generation algorithm using the L-BFGS as the optimization algorithm.

The “prefer_img_dgn_gd” implements the preferred image generation algorithm using a deep generator network (DGN) to introduce image prior, and using gradient descent (GD) with momentum as the optimization algorithm.

The “prefer_img_dgn_lbfgs” implements the preferred image generation algorithm using a deep generator network (DGN) to introduce image prior, and using the L-BFGS as the optimization algorithm.


## Basic Usage

### Basic usage of “prefer_img_gd” and “prefer_img_lbfgs”:

``` python
from cnnpref.prefer_img_gd (or cnnpref.prefer_img_lbfgs) import generate_image
img = generate_image(net, layer, feature_mask)
```

- Input
    - `net`: CNN model (caffe.Classifier or caffe.Net object).
    - `layer`: The name of the layer for the target units (str).
    - `feature_mask`: The mask used to select the target units. The shape of the mask should be the same as that of the CNN features in that layer. The values of the mask array are binary, (1: target uint; 0: irrelevant unit). (ndarray)
- Output
    - `img`: the preferred image (numpy.float32 [227x227x3])

### Basic usage of “prefer_img_dgn_gd” and “prefer_img_dgn_lbfgs”:

``` python
from cnnpref.prefer_img_dgn_gd (or cnnpref.prefer_img_dgn_lbfgs) import generate_image
img = generate_image(net_gen, net, layer, feature_mask)
```

- Input
    - `net_gen`: Deep generator net (caffe.Net object).
    - `net`: CNN model (caffe.Classifier or caffe.Net object).
    - `layer`: The name of the layer for the target units (str).
    - `feature_mask`: The mask used to select the target units. The shape of the mask should be the same as that of the CNN features in that layer. The values of the mask array are binary, (1: target uint; 0: irrelevant unit). (ndarray)
- Output
    - `img`: the preferred image (numpy.float32 [227x227x3])


## Example Codes

### Example codes to generate preferred image for the units in lower layers:

- generate_preferred_image_gd__for_lower_layer.py
- generate_preferred_image_lbfgs__for_lower_layer.py
- generate_preferred_image_dgn_gd__for_lower_layer.py
- generate_preferred_image_dgn_lbfgs__for_lower_layer.py

### Example codes to generate preferred image for the units in higher layers:

- generate_preferred_image_gd__for_higher_layer.py
- generate_preferred_image_lbfgs__for_higher_layer.py
- generate_preferred_image_dgn_gd__for_higher_layer.py
- generate_preferred_image_dgn_lbfgs__for_higher_layer.py


## CNN Model

In the example codes, we use pre-trained AlexNet (caffemodel_url: http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel).
You can replace it with any other CNN models in the example codes.
In order to make back-propagation work, one line should be added to the prototxt file (the file describes the configuration of the CNN model: https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt):

`force_backward: true`.


## CNN features before or after ReLU

In the example codes in this repository, we define CNN features of conv layers or fc layers as the output immediately after the convolutional or fully-connected computation, before applying the Rectified-Linear-Unit (ReLU).
However, as default setting, ReLU operation is an in-place computation, which will override the CNN features we need.
In order to use the CNN features before the ReLU operation, we need to modify the prototxt file.
Taking the AlexNet prototxt file as an example:

In the original prototxt file, ReLU is in-place computation:

```
layer {
  name: "relu1"
  type: "RELU"
  bottom: "conv1"
  top: "conv1"
}
```

Now, we modify it as:

```
layer {
  name: "relu1"
  type: "RELU"
  bottom: "conv1"
  top: "relu1"
}
```

## Deep Generator Network

In the example codes in this repository, we use pre-trained deep generator network (DGN) from the study [4] (downloaded from: https://lmb.informatik.uni-freiburg.de/resources/binaries/arxiv2016_alexnet_inversion_with_gans/release_deepsim_v0.zip).
In order to make back-propagation work, one line should be added to the prototxt file (the file describes the configuration of the DGN):

`force_backward: true`.


## Speed generation of preferred image
The preferred image generation is designed to generate one preferred image each time, while the CNN model and deep generator net process batch of data.
In order to avoid irrelevant calculation and speed the image generation process, we can modify the batch size to  be 1.
For example, we can set the first dimension (batch size) to 1 in the prototxt of the deep generator net (/net/generator_for_inverting_fc6/generator.prototxt):

```
...
input: "feat"
input_shape {
  dim: 1  # 64 --> 1
  dim: 4096
}

...

layer {
  name: "reshape_relu_defc5"
  type: "Reshape"
  bottom: "relu_defc5"
  top: "reshape_relu_defc5"
  reshape_param {
    shape { 
      dim: 1  # 64 --> 1
      dim: 256
      dim: 4
      dim: 4
    }
  }
}
...

```

## Reference
- [1] Karen Simonyan, Andrea Vedaldi, Andrew Zisserman (2013). Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps. https://arxiv.org/abs/1312.6034
- [2] Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, Hod Lipson (2015). Understanding Neural Networks Through Deep Visualization. https://arxiv.org/abs/1506.06579
- [3] Nguyen A, Dosovitskiy A, Yosinski J, Brox T, and Clune J (2016). Synthesizing the preferred inputs for neurons in neural networks via deep generator networks. https://arxiv.org/abs/1605.09304
- [4] Dosovitskiy A and Brox T (2016). Generating Images with Perceptual Similarity Metrics based on Deep Networks. https://arxiv.org/abs/1602.02644


## Copyright and License

Copyright (c) 2018 Kamitani Lab (http://kamitani-lab.ist.i.kyoto-u.ac.jp/)

The scripts provided here are released under the MIT license (http://opensource.org/licenses/mit-license.php).


## Authors

Shen Guo-Hua (E-mail: shen-gh@atr.jp)


## Acknowledgement

The author thanks Mitsuaki Tsukamoto for software installation and computational environment setting.
The author thanks precious discussion and advice from the members in DNI (http://www.cns.atr.jp/dni/) and Kamitani Lab (http://kamitani-lab.ist.i.kyoto-u.ac.jp/).

The codes in this repository are inspired by many existing image generation or synthesizing studies and their open-source implementations, including:

- The source code of the paper “Understanding deep image representations by inverting them”: https://github.com/aravindhm/deep-goggle
- The source code of the paper “Synthesizing the preferred inputs for neurons in neural networks via deep generator networks”: https://github.com/Evolving-AI-Lab/synthesizing
- Deepdream: https://github.com/google/deepdream
- Deepdraw: https://www.auduno.com/2015/07/29/visualizing-googlenet-classes/
- Several tricks (e.g. clipping the pixels with small norm, and clipping the pixels with small contribution) in “iCNN_GD” are borrowed from the paper “Understanding Neural Networks Through Deep Visualization”: http://yosinski.com/deepvis
