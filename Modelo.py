import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import os.path as path

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.models import model_from_json
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

np.random.seed(5776)

IMAGE_SIZE = 128
IMAGE_NUM_CHANNELS = 1
clases = ['Class1.1', 'Class1.2', 'Class1.3', 'Class2.2', 'Class2.2', 'Class3.1', 'Class3.2', 'Class4.1', 'Class4.2',
          'Class5.1', 'Class5.2', 'Class5.3', 'Class5.4', 'Class6.1', 'Class6.2', 'Class7.1', 'Class7.2', 'Class7.3',
          'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7', 'Class9.1', 'Class9.2',
          'Class9.3', 'Class10.1', 'Class10.2', 'Class10.3', 'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4',
          'Class11.5', 'Class11.6']

def networkArchitecture():
    print ("Arquitectura de la CNN")
    model = Sequential()

    model.add(Conv2D(80, (5, 5), activation='relu', input_shape=(IMAGE_NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(96, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(192, 3, 3, activation='relu'))

    model.add(Convolution2D(192, 3, 3, activation='relu'))

    model.add(Convolution2D(384, 3, 3, activation='relu'))

    model.add(Convolution2D(384, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Dense(37, activation='softmax'))

    return model

def trainingModelo():
    print("Entrenando el model ...")
    #obtenemos los valores guardados de las imagenes de train que provienen del pickle
    pickle_x = open("./pickles/X.pickle", "rb")
    X = pickle.load(pickle_x)
    # obtenemos los valores guardados del csv de train que provienen del pickle
    pickle_y = open("./pickles/y.pickle", "rb")
    y = pickle.load(pickle_y)

    X = X / 255.0

    #dividimos el dataset en entrenamiento (90%) y validacion(90%)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1, random_state=42)

    X_train = X_train.reshape(X_train.shape[0], IMAGE_NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    X_validation = X_validation.reshape(X_validation.shape[0], IMAGE_NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

    model = networkArchitecture()
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=3, validation_split=0.3)

    #hacemos la predicciones sobre los datos de validacion
    predicciones_val = model.predict(X_validation)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Pesos del modelo guardados.")

    print("Finalizacion de entrenamiento.")
    return model

def loadModeltesting():

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("model.h5")
    print("Modelo cargado desde el archivo")
    return loaded_model

def testingModelo():

    #obtenemos la ruta donde se encuntran el csv de las imagenes de test
    test_data_path = './csv/testGalaxyZoo.csv'
    test_data = pd.read_csv(test_data_path)

    # obtenemos los valores guardados de las imagenes de test que provienen del pickle
    pickle_x_test = open("./pickles/X_test.pickle", "rb")
    X_t = pickle.load(pickle_x_test)

    X_t = X_t / 255.0

    X_test = X_t.reshape(X_t.shape[0], IMAGE_NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    print("Inicializa el testing ...")
    #Predecimos los valores de las imagenes de test
    if path.exists('model.json'):
        model=loadModeltesting()
    else:
        model = trainingModelo()
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    predicciones_test = model.predict(X_test)

    #print(predicciones_test)

    #Guardamos y generamos el csv
    output = pd.DataFrame(data=np.array(predicciones_test), columns=clases)
    output.insert(loc=0, column='GalaxyID', value=test_data.GalaxyID)
    output.to_csv('csv/galaxyPredictions.csv', index=False)

    print("Finalizacion del testing.")

def main():
    testingModelo()

if __name__ == '__main__':
    main()


