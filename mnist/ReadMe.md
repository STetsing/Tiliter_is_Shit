# Mnist hand written digits recognition

This directory presents a simple solution of solving the mnist problem on a custum data set using deep neural networks with state of the art accuracy. Only images in the train directories have been used in order to train the network, which is then splitted for validating the nework.

This neural network architecture is rather very simple, uses the relu activation and the dropout regularisation. Due to the size of the original image, thge 2x2 pooling can only be carried out twice without too much coding efforts. The cross entropy is used as we deal with a binary problem. For expriming the answer in classes, i apply at  the network output an softmax activation. The  implementation is rather 'old school'. Existing libraries may achieve the presented results in much lesser code. I use purposely tensorflow with basics operations.

The achieved accuracy carried out on the test images after training is: 0.99225

Used learning rate: 3e-05

## Prerequisites
The folder 'data' must be in the same directory as the scripts


## Train
Run
```bash
python train_mnist.py -e 1e-5
```
for training the network with a learnin rate of epsilon =1e-5

## Test
Run
```bash
python test_mnist.py
```
for testing the test images. These are read automatically from the test directories

