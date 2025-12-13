import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


input_layer = Input(shape=(28, 28), name="entrada")
flatten_layer = Flatten()(input_layer)

hidden1 = Dense(128, activation='relu', name="oculta_1")(flatten_layer)

hidden2 = Dense(64, activation='relu', name="oculta_2")(hidden1)

output_layer = Dense(10, activation='softmax', name="saida")(hidden2)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Treinando modelo mais profundo...")
model.fit(x_train, y_train, epochs=7)

model.save('modelo_visual.h5')
print("Modelo salvo como 'modelo_visual.h5'")