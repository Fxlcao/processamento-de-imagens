import sys
import logging
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Ajustando a codificação do sistema para UTF-8
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(encoding='utf-8')

# Configurando o backend do Matplotlib
import matplotlib
matplotlib.use('Agg')  # Usando backend sem interface gráfica

# Carregar e preparar os dados do CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizar as imagens para valores entre 0 e 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Converter as labels para o formato one-hot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
print(f"x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}")

# Aplicar sharpening às imagens
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype='float32')

x_train_sharpened = np.array([cv2.filter2D(img, -1, sharpen_kernel) for img in x_train])
x_test_sharpened = np.array([cv2.filter2D(img, -1, sharpen_kernel) for img in x_test])

# Aumentar as imagens em 2x
x_train_upscaled = np.array([cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC) for img in x_train_sharpened])
x_test_upscaled = np.array([cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC) for img in x_test_sharpened])

print(f"x_train_upscaled.shape: {x_train_upscaled.shape}, x_test_upscaled.shape: {x_test_upscaled.shape}")

# Construir o modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo com imagens ampliadas
history = model.fit(x_train_upscaled, y_train, epochs=25, batch_size=32, validation_data=(x_test_upscaled, y_test))

# Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test_upscaled, y_test)
print(f"Test accuracy: {test_acc}")

# Exibir as primeiras 5 imagens de treino ampliadas e suas respectivas classes
for i in range(5):
    plt.imshow(x_train_upscaled[i])  # Exibe a imagem ampliada
    plt.title(f"Classe: {y_train[i].argmax()}")  # Mostra a classe (rótulo)
    plt.savefig(f"imagem_{i}.png")  # Salva a imagem
    plt.close()

# Exibir os gráficos de treinamento e validação
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia do Modelo')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend(loc='lower right')
plt.savefig("grafico_acuracia.png")
plt.close()

plt.plot(history.history['loss'], label='Perda do Treinamento')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Perda do Modelo')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend(loc='upper right')
plt.savefig("grafico_perda.png")
plt.close()
