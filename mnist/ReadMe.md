# Mnist hand written digits recognition

This directory presents a simple solution of solving the mnist problem using deep neural networks with state of the art accuracy. Only images in the train directories have been used in order to train the network, which is then splitted for validating the nework.

The achieved accuracy carried out on the test images after training is:
used learning rate: 3e-05

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

