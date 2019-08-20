# pixel-cnn++

This is a fork of: https://github.com/openai/pixel-cnn

The following enhancements are added:

## 1. Support for MNIST dataset

The original code supports CIFAR-10 and Small Imagenet. Support for MNIST is added. 

Use the following command line option to use MNIST:
```
--data_set='mnist'
```

The MNIST support is adapted from a PyTorch implementation of PixelCNN++: https://github.com/pclucas14/pixel-cnn-pp


## 2. CoordConv

Based on the paper:
["An intriguing failing of convolutional neural networks and the CoordConv solution"](https://arxiv.org/pdf/1807.03247)

Use the following command line option to add CoordConv channels:
```
-ac
```

The code for this adapted from section S9 of the paper

## 3. Non-Local Neural Networks

Based on the paper:
["Non-local Neural Networks"](https://arxiv.org/abs/1711.07971)

Use the following command line option to add CoordConv channels:
```
-nl
```

The code for this is adapted from https://github.com/lucasb-eyer/nonlocal-tf

## 4. Focal Loss for Dense Object Detection

Based on the paper:
["Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002)

Use the following command line option to add CoordConv channels:
```
-fl
```

## 5. Original Text 

From: https://github.com/openai/pixel-cnn


This is a Python3 / [Tensorflow](https://www.tensorflow.org/) implementation 
of [PixelCNN++](https://openreview.net/pdf?id=BJrFC6ceg), as described in the following 
paper:

**PixelCNN++: A PixelCNN Implementation with Discretized Logistic Mixture Likelihood and Other Modifications**, by
Tim Salimans, Andrej Karpathy, Xi Chen, Diederik P. Kingma, and Yaroslav Bulatov.

Our work builds on PixelCNNs that were originally proposed in [van der Oord et al.](https://arxiv.org/abs/1606.05328) 
in June 2016. PixelCNNs are a class of powerful generative models with tractable 
likelihood that are also easy to sample from. The core convolutional neural network
computes a probability distribution over a value of one pixel conditioned on the values
of pixels to the left and above it. Below are example samples from a model
trained on CIFAR-10 that achieves **2.92 bits per dimension** (compared to 3.03 of 
the PixelCNN in van der Oord et al.):

Samples from the model (**left**) and samples from a model that is conditioned
on the CIFAR-10 class labels (**right**):

![Improved PixelCNN papers](data/pixelcnn_samples.png)

This code supports multi-GPU training of our improved PixelCNN on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
and [Small ImageNet](http://image-net.org/small/download.php), but is easy to adapt
for additional datasets. Training on a machine with 8 Maxwell TITAN X GPUs achieves
3.0 bits per dimension in about 10 hours and it takes approximately 5 days to converge to 2.92.

## Setup

To run this code you need the following:

- a machine with multiple GPUs
- Python3
- Numpy, TensorFlow and imageio packages:
```
pip install numpy tensorflow-gpu imageio
```

## Training the model

Use the `train.py` script to train the model. To train the default model on 
CIFAR-10 simply use:

```
python3 train.py
```

You might want to at least change the `--data_dir` and `--save_dir` which
point to paths on your system to download the data to (if not available), and
where to save the checkpoints.

**I want to train on fewer GPUs**. To train on fewer GPUs we recommend using `CUDA_VISIBLE_DEVICES` 
to narrow the visibility of GPUs to only a few and then run the script. Don't forget to modulate
the flag `--nr_gpu` accordingly.

**I want to train on my own dataset**. Have a look at the `DataLoader` classes
in the `data/` folder. You have to write an analogous data iterator object for
your own dataset and the code should work well from there.

## Pretrained model checkpoint

You can download our pretrained (TensorFlow) model that achieves 2.92 bpd on CIFAR-10 [here](http://alpha.openai.com/pxpp.zip) (656MB).

## Citation

If you find this code useful please cite us in your work:

```
@inproceedings{Salimans2017PixeCNN,
  title={PixelCNN++: A PixelCNN Implementation with Discretized Logistic Mixture Likelihood and Other Modifications},
  author={Tim Salimans and Andrej Karpathy and Xi Chen and Diederik P. Kingma},
  booktitle={ICLR},
  year={2017}
}
```
