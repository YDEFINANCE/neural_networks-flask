import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix



###########################################
#### Préparation de la base de données ####
###########################################


#Importation de la base de données CIFAR10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Encodage one-hot #
y_train_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)

# Les chiffres représentant les images compris entre 0 et 1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255


######################################
### réseau de neurones convolutif  ###
######################################
keras.backend.clear_session()
model4 = keras.models.Sequential()

model4.add(Conv2D(32,(3,3),padding="same", input_shape=(32,32,3),activation='relu'))
model4.add(Conv2D(32,(3,3),activation='relu'))
model4.add(MaxPooling2D(pool_size=(2,2)))
model4.add(Dropout(0.25))

model4.add(Conv2D(64,(3,3),padding="same",activation='relu'))
model4.add(Conv2D(64,(3,3),activation='relu'))
model4.add(MaxPooling2D(pool_size=(2,2)))
model4.add(Dropout(0.25))

model4.add(Flatten())
model4.add(Dense(512,activation='relu'))
model4.add(Dropout(0.5))
model4.add(Dense(10,activation='softmax'))

early_stop = EarlyStopping(monitor='val_loss',patience=2)

model4.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model4.summary()

model4.fit(x_train,
           y_train,
           batch_size=32, 
           validation_data=(x_test, y_test),
           epochs=50, 
           callbacks=[early_stop])

losses = pd.DataFrame(model4.history.history)
losses[['accuracy', 'val_accuracy']].plot()
losses[['loss', 'val_loss']].plot()

pred = model4.predict(x_test)
pred = pred.argmax(axis=-1)

model4.save('cnn_cifar10_4.h5')

######################################
############  résultats ##############
######################################
nom_objet = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
plot_confusion_matrix(confusion_matrix(y_test, pred),figsize=(12,12),class_names=nom_objet, show_normed=True)