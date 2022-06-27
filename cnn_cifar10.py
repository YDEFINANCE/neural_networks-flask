######################################
##### Importation des librairies #####
######################################

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from keras.models import load_model
from skimage.transform import resize



####################################################
### Importation du réseau de neurones à utiliser ###
####################################################
#image="deer.jpg"
cnn='cnn_cifar10.h5'

def run_model_cnn_cifar10(image,cnn):
    model = load_model(cnn)
    my_image = plt.imread(image)
    my_image = resize(my_image, (32,32,3))
    labels =  ['Avion', 'Automobile', 'Oiseau', 'Chat', 'Cerf', 'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion']
    probabilities = model.predict(np.array( [my_image,] ))
    print(probabilities)
    index = np.argsort(probabilities[0,:])
    print(index)
    res=labels[index[9]]
    return res

#resultat=run_model_cnn_cifar10(image)
#print(resultat)
