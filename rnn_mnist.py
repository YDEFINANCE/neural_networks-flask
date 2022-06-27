import numpy as np
import matplotlib.pyplot as plt
from random import choice
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from skimage.transform import resize


########################################
### Téléchargement du jeu de données ###
########################################
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train de taille (60000*28*28)
#x_test de taille (10000*28*28)
#y_train de taille (60000,1)
#y_test de taille (10000,1)



########################################
### Initialisation du jeu de données ###
########################################
x_train = x_train / 255 
x_test = x_test / 255
# Division des pixels des images par 255 de manière à ce que chaque pixel représenté numériquement soit compris entre 0 et 1
#One hot encoding
y_train = to_categorical(y_train) #y_train de taille (60000,10)
y_test = to_categorical(y_test) #y_test de taille (10000,10)


entrees=x_train
sorties=y_train
taille_batch = 100
a= (-taille_batch)
variables = np.zeros((784,10)) #784 variables pour chaque chiffre de 1 à 10, 784=28*28, nb de pixels par image


for i in range (200):
    #Création du batch
    a=(a+taille_batch)%(len(entrees)-taille_batch)
    x_batch, y_batch = np.array(entrees[a:a+taille_batch]), np.array(sorties[a:a+taille_batch])
    x_batch=x_batch.reshape(100,784)
    x_batch=x_batch.astype(np.float)
    #Apprendre le batch
    derreur_dx = 2*np.dot(x_batch.T,(np.dot(x_batch,variables)-y_batch))
    variables -= (derreur_dx/len(x_batch))*0.02

print(np.shape(variables))


#image="img_3.jpg"

def run_model_rnn_mnist(image):
    my_image = plt.imread(image)
    my_image = resize(my_image, (28,28,1))
    image = my_image.reshape(1,784)
    res = np.argmax(np.dot(image, variables))
    return res

#res=run_model_rnn_mnist(image)
#print("C'est un " + str(res))

