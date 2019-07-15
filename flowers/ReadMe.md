# Flowers Classification
 The goal of this task is to create an optimal model for classifying the flowers present in the execution directory. For this purpose i used the random gridsearcher on the c-support vector machine. I have also tried out random decision forest, which had apparently nearly the same test and training accuracy. As i analize the result i have construted, there might be either a bug somehow in the preprocessing stage, or the problem is to hard to be solved using the approach i have chosen. This explains why i apply the kernel PCA for dimensionality reduction, where i project the data on the first 50 eigenvectors. Previous runs on the dataset using the classical PCA have revealed, that the information ratio is over 75%.
 I observed the train accuracy: 0.8415268941584731 and the test accuracy: 0.5421965317919075. Projecting the data on more eigenvectors (as 50) does indeed increase the training accuracy, but the test accuracy decreases. Hence overfitting.

## Prerequisites
The folder 'flowers' must be in the same directory as the scripts

## Train
Run
```bash
python train_flower.py -s -n <eps>
```
where you save the model with the flag -s and submit the number fold for the cross validation with -n

Example
```bash
python train_flower.py -s -n 20

```

## Test
Run
```bash
python test_flowers_dir.py -d <dir>
```
for batch testing on a directory

Example
```bash
python test_flowers_dir.py -d  flowers/daisy/
```

or
```bash
python test_flowers_img.py -i <img_file>
```
for testing a single image

Example
```bash
python test_flowers_img.py -i flowers/dandelion/11595255065_d9550012fc.jpg
```
