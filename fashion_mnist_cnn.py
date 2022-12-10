import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#fashion_mnist_train_data = pd.read
fashion_mnist_train_data = pd.read_csv("fashion-mnist_train.csv")
fashion_mnist_test_data = pd.read_csv("fashion-mnist_test.csv")

fashion_mnist_train_data.head()

fashion_mnist_test_data.head()

#train data split
xtrain = fashion_mnist_train_data.drop(['label'], axis = 1)
ytrain = fashion_mnist_train_data['label']

#test data split
xtest = fashion_mnist_test_data.drop(['label'], axis = 1)
ytest = fashion_mnist_test_data['label']

#print(xtest.head())
#print(ytest.head)

#assigning labels to the images as per the dataset
accessory_label_dict = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 
                     3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 
                     7:'Sneaker', 8:'Bag', 9:'Ankle boot'}

def accessory_label(x):
    return accessory_label_dict[x]

print('Training data shape : ', xtrain.shape, ytrain.shape)

print('Testing data shape : ', xtest.shape, ytest.shape)

xtrain.isnull().any().describe()

"""#normalization of datasets
xtrain = xtrain / 255.
xtest = xtest / 255.

####Here we use 28 x 28 because "Each image is a standardized 28×28 size in grayscale (784 total pixels)" according to the dataset's description
"""

#displaying the image for train set
train_number = 1
plt.imshow(xtrain.values.reshape(-1,28,28,1)[train_number][:,:,0])
print('This represent a : ' + accessory_label(ytrain[train_number]) + " from the training data")

#displaying the image for test set
test_example = 0
plt.imshow(xtest.values.reshape(-1,28,28,1)[test_example][:,:,0])
print('This is a : ' + accessory_label(ytest[test_example]) + " from the test data")