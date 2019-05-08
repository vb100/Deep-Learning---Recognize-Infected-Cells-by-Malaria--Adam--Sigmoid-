The project contain folders for labeled infected and non-infected cells by malaria. 
The code define an 4 layers ANN that learning to recognize infected cells by a given 
shuffled examples from working directory and provide a learning curve as output.
For optimization Adam and MiniBatch methods have been applied (Tensorflow). 

First of all, after all images are loaded, every single one is converted from RGB to grayscale,
then transformed to the same resolution 134/2 × 134/2 and converting to 2D Numpy arrays. Then all
data manipulations are being performing through algoritm of splitting data to train and test sets
by a given proportion and Deep Learning part. 

For this ANN I used structure Linear->ReLU->Linear->ReLU->Linear->ReLU->Linear->Sigmoid for binary classification.

For optimization I decided to apply Adam and MiniBatch methods.

In total more than 27k examples has been included into the model which has been splitted to train and test sets.
Number of nodes for each hidden layers are: 30, 15, 7 and 1 (W1–W4).

To train a model takes appr. 10 minutes on my laptop.