import numpy as np
import pandas as pd
import os
import time

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from tensorflow.keras.regularizers import l2
from keras.models import Sequential


###########################################
#### Préparation de la base de données ####
###########################################

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

metadata_path = './cifar-100-python/meta' 
metadata = unpickle(metadata_path)
superclass_dict = dict(list(enumerate(metadata[b'coarse_label_names'])))


data_pre_path = './cifar-100-python/' 
# File paths
data_train_path = data_pre_path + 'train'
data_test_path = data_pre_path + 'test'
# Read dictionary
data_train_dict = unpickle(data_train_path)
data_test_dict = unpickle(data_test_path)

data_train = data_train_dict[b'data']
label_train = np.array(data_train_dict[b'coarse_labels'])
data_test = data_test_dict[b'data']
label_test = np.array(data_test_dict[b'coarse_labels'])

x_train = data_train.astype('float32')
x_test = data_test.astype('float32')

x_train = x_train.reshape(len(x_train),3,32,32).transpose(0,2,3,1)
x_test = x_test.reshape(len(x_test),3,32,32).transpose(0,2,3,1)
x_train = x_train / 255
x_test = x_test / 255
y_test = label_test
y_train = label_train

######################################
### réseau de neurones convolutif  ###
######################################

model = Sequential()
 
######     1ère couche    #####
model.add(Conv2D(256,(3,3),padding='same',input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
######     2nd couche    ######
model.add(Conv2D(512,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


######     3 ème couche    ######
model.add(Conv2D(512,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

######     4 ème couche    ######
model.add(Conv2D(512,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization(momentum=0.95, 
        epsilon=0.005,
        beta_initializer=RandomNormal(mean=0.0, stddev=0.05), 
        gamma_initializer=Constant(value=0.9)))

model.add(Dense(100,activation='softmax'))

model.summary()

# Data Augmentation
# Adding data augmentation for creating more images
# Configuration for creating new images
train_datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

training_start = time.time()
hist = model.fit(x_train, y_train, 
           batch_size=64, epochs=20, 
           validation_data=(x_test, y_test),
           validation_split=0.2)
training_stop = time.time()
training_time = training_stop - training_start
print(f"Training time: {training_time}")

losses = pd.DataFrame(model.history.history)
losses[['accuracy', 'val_accuracy']].plot()
losses[['loss', 'val_loss']].plot()

pred = model.predict(x_test)
pred = pred.argmax(axis=-1)
model.save('cnn_cifar100_7.h5')


######################################
############  résultats ##############
######################################
nom_objet = ["aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables", "household electrical devices", "household furniture", "insects", "large carnivores", "large man-made outdoor things","large natural outdoor scenes","large omnivores and herbivores","medium-sized mammals","non-insect invertebrates","people","reptiles","small mammals","trees","vehicles 1","vehicles 2" ]
plot_confusion_matrix(confusion_matrix(y_test, pred),figsize=(12,12),class_names=nom_objet, show_normed=True)

