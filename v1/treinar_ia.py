import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

#carregar os dados // load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalizar os dados // normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

#criar modelo de rede neural // create neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)), #transrtoma a matriz 28x28 em uma linha // transform the 28x28 matrix into a line
    Dense(128, activation='relu'), #camada oculta com 128 neuronios // hidden layer with 128 neurons
    Dense(10, activation='softmax') #saida (0-9) // output (0-9)
])

#compilar e treinar // compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Treinando a IA... // Training AI...")
model.fit(x_train, y_train, epochs=5) #5 rodadas de treino // 5 rounds of training

#salvar o cerebro treinado // save training model
model.save('model.h5')
print(f"Modelo salvo como: {model.save} // Model save as: {model.save}")