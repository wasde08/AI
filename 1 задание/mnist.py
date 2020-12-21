from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from PIL import Image
import time
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd



def prRes(str1):
  list_of_files = os.listdir(str1)
  for file in list_of_files:
      image_file_name = os.path.join(str1, file)
      if ".jpg" in image_file_name:
          print("Должно быть:" +str(image_file_name.split("content/")[1].split("/")[1].split(".")[0])+ " -  результат: " + str(mlp_digits_predict(model, image_file_name)))
        
def load_images_to_data(image_label, image_directory, features_data, label_data):
    list_of_files = os.listdir(image_directory)
    for file in list_of_files:
        image_file_name = os.path.join(image_directory, file)
        if ".png" in image_file_name:
            img = Image.open(image_file_name).convert("L")
            img = np.resize(img, (28,28,1))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1, 28, 28, 1)
            features_data = np.append(features_data, im2arr, axis=0)
            label_data = np.append(label_data, [image_label], axis=0)
    return features_data, label_data

def mlp_digits_predict(model, image_file):
   image_size = 28
   img = keras.preprocessing.image.load_img(image_file, target_size=(image_size, image_size), color_mode='grayscale')
   img_arr = np.expand_dims(img, axis=0)
   img_arr = 1 - img_arr/255.0
   img_arr = img_arr.reshape((1, 28, 28, 1))
   result = model.predict_classes([img_arr])
   return result[0]

def mnist_make_model(image_w: int, image_h: int):  
   image_size = 28
   num_channels = 1  # 1 for grayscale images
   num_classes = 10   
   model = Sequential()
   model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same',
                    input_shape=(image_size, image_size, num_channels)))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',padding='same'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',padding='same'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Flatten())
   # Densely connected layers
   model.add(Dense(128, activation='relu'))
   # Output layer
   model.add(Dense(num_classes, activation='softmax'))
   model.compile(optimizer=Adam(), loss='categorical_crossentropy',metrics=['accuracy'])
   return model

def mnist_mlp_train(model):
    image_size = 28
    num_channels = 1
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train,(x_train.shape[0], image_size, image_size, num_channels))
    x_test = np.reshape(x_test,(x_test.shape[0], image_size, image_size, num_channels))
    (x_train, y_train) = load_images_to_data('1', '/content/mytrainimg/1', x_train, y_train)
    #(x_test, y_test) = load_images_to_data('1', '/content/mytestimg/1', x_test, y_test)
    x_train = x_train.astype('float32')/255.0
    x_test = x_test.astype('float32')/255.0
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model.fit(x_train, y_train, epochs=40, batch_size=32)

def confusion_matrix(model):
    image_size = 28
    num_channels = 1
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train,(x_train.shape[0], image_size, image_size, num_channels))
    x_test = np.reshape(x_test,(x_test.shape[0], image_size, image_size, num_channels))
    x_train = x_train.astype('float32')/255.0
    x_test = x_test.astype('float32')/255.0
    classes = [0,1,2,3,4,5,6,7,8,9]
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

'''
model = mnist_make_model(image_w=28, image_h=28)
mnist_mlp_train(model)
model.save('mlp_digits_28x28.h5')  
'''  


p = os.path.abspath('/content/mlp_digits_28x28.h5')
model = tf.keras.models.load_model(p)

confusion_matrix(model)

image_directory = "/content/img/"
prRes(image_directory)
print()

print("my data")
image_directory1 = "/content/mytestimg/1/"
prRes(image_directory1)



