import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
pasta_models = os.path.join(diretorio_atual, '..', 'models')
caminho_arquivo_modelo = os.path.join(pasta_models, 'modelo_visual.h5')

if not os.path.exists(pasta_models): os.makedirs(pasta_models)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

datagen = ImageDataGenerator(
    rotation_range=15,      
    zoom_range=0.15,        
    width_shift_range=0.15, 
    height_shift_range=0.15 
)
datagen.fit(x_train)

input_layer = Input(shape=(28, 28, 1), name="entrada")
flatten_layer = Flatten()(input_layer)

hidden1 = Dense(256, activation='relu', name="oculta_1")(flatten_layer)
dropout1 = Dropout(0.2)(hidden1) 

hidden2 = Dense(128, activation='relu', name="oculta_2")(dropout1)
dropout2 = Dropout(0.2)(hidden2)

output_layer = Dense(10, activation='softmax', name="saida")(dropout2)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Iniciando Treino Turbo (Isso pode demorar 1 ou 2 minutos)...")
model.fit(datagen.flow(x_train, y_train, batch_size=32),
          epochs=50, 
          validation_data=(x_test, y_test))

model.save(caminho_arquivo_modelo)
print(f"Modelo Turbo salvo em: {caminho_arquivo_modelo}")