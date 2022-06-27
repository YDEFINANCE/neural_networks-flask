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
#image="bicycle.jpg"
cnn='cnn_cifar100.h5'

def run_model_cnn_cifar100(image,cnn):
    model = load_model(cnn)
    my_image = plt.imread(image)
    my_image = resize(my_image, (32,32,3))
    labels =  ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 
           'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 
           'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 
           'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 
           'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 
           'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 
           'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 
           'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 
           'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    probabilities = model.predict(np.array( [my_image,] ))
    print(probabilities)
    index = np.argsort(probabilities[0,:])
    print(index)
    res=labels[index[99]]
    return res

#resultat=run_model_cnn_cifar100(image)
#print(resultat)






