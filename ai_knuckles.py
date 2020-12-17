import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf
#from google.colab import drive
# Accessing My Google Drive
#drive.mount('/content/drive')


def prRes(str1):
  list_of_files = os.listdir(str1)
  for file in list_of_files:
      image_file_name = os.path.join(str1, file)
      if ".png" in image_file_name:
          print("Должно быть:" +str(image_file_name)+ " -  результат: " + str(mlp_digits_predict(model, image_file_name)))

def mlp_digits_predict(model, image_file):
   image_size = 160
   img = keras.preprocessing.image.load_img(image_file, target_size=(image_size, image_size), color_mode='rgb')
   img_arr = np.expand_dims(img, axis=0)
   img_arr = 1 - img_arr/255.0
   img_arr = img_arr.reshape((1, 160, 160, 3))
   result = model.predict([img_arr])
   return result[0]

image_path = "/content/drive/MyDrive/knuck/kn/"
'''
i = 1
list_of_files = os.listdir(image_path)
for file in list_of_files:
    image_file_name = os.path.join(image_path, file)
    if ".png" in image_file_name:
      newName = str(i)+".png"
      os.rename(file,newName)
      i+=1
'''
# Каталог с данными для обучения
train_dir = '/content/drive/MyDrive/knuck/kn/'
# Каталог с данными для проверки
val_dir = '/content/drive/MyDrive/testkn/'
# Каталог с данными для тестирования
test_dir = '/content/drive/MyDrive/testkn/'
# Размеры изображения
img_width, img_height = 160, 160
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 10
# Размер мини-выборки
batch_size = 16
# Количество изображений для обучения
nb_train_samples = 841
# Количество изображений для проверки
nb_validation_samples = 28
# Количество изображений для тестирования
nb_test_samples = 28
num_classes = 4

def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

def model_train(model):
    datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size)

#model = create_model()
#model_train(model)
#model.save('mlp_digits_160x160.h5') 

p = os.path.abspath('/content/mlp_digits_160x160.h5')
model = tf.keras.models.load_model(p) 


#scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)

#print(scores)
print("-----")
#image_directory = "/content/test/"
#prRes(image_directory)
'''
import cv2
import numpy as np

image = cv2.imread('/content/test/02.png')
image = cv2.resize(image, (160, 160))
image = image.astype('float32')
image = image.reshape((1,160, 160, 3))
image = 255-image
image /= 255

predict = model.predict(image)
plt.imshow(image.reshape(160, 160,3))
print(predict, predict.argmax())



image_file_name = os.path.join("/content/test/02.png")
print(mlp_digits_predict(model, image_file_name))
print("---")
image_file_name1 = os.path.join("/content/test/301.png")
print(mlp_digits_predict(model, image_file_name1))
#print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))
'''

      
