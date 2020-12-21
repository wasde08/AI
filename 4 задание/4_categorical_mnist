from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from PIL import Image
import time
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def confusion_matrix(model):
    image_size = 28
    num_channels = 1
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    train_filter = np.where((y_train == 0)|(y_train == 1)|(y_train == 2)|(y_train == 3))
    test_filter = np.where((y_test == 0)|(y_test == 1)|(y_test == 2)|(y_test == 3))
    x_train, y_train = x_train[train_filter], y_train[train_filter]
    x_test, y_test = x_test[test_filter], y_test[test_filter]
    x_train = x_train.reshape(len(x_train),28,28,1)
    x_test = x_test.reshape(len(x_test),28,28,1)
    x_train = x_train.astype('float32')/255.0
    x_test = x_test.astype('float32')/255.0
    classes = [0,1,2,3]
    y_pred=model.predict_classes(x_test)
    con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm,
                        index = classes, 
                        columns = classes)
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def get_model():
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1)))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(4, activation='softmax'))

  model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['accuracy'])
  
  return model


def model_train(model):
    image_size = 28
    num_channels = 1
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    train_filter = np.where((y_train == 0)|(y_train == 1)|(y_train == 2)|(y_train == 3))
    test_filter = np.where((y_test == 0)|(y_test == 1)|(y_test == 2)|(y_test == 3))
    x_train, y_train = x_train[train_filter], y_train[train_filter]
    x_test, y_test = x_test[test_filter], y_test[test_filter]
    
    x_train = x_train.reshape(len(x_train),28,28,1)
    x_test = x_test.reshape(len(x_test),28,28,1)

    x_train = x_train.astype('float32')/255.0
    x_test = x_test.astype('float32')/255.0

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    
    hist = model.fit(x_train, y_train,validation_data=(x_test, y_test),epochs=1)
    print(hist.history)   
   
    
    

model = get_model()
model_train(model)
model.save('4_cat_28x28.h5')
confusion_matrix(model)
