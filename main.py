# import numpy as np
# import cv2
# from cv2 import dnn_superres
# import matplotlib.pyplot as plt
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras import layers, models
# import tensorflow as tf

# # Carregar o dataset CIFAR-10
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# # Exibir as dimensões dos dados
# print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
# print(f"x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}")

# # Criar objeto SR para upscaling
# print("Criando o objeto SR")
# sr = dnn_superres.DnnSuperResImpl_create()
# model_path = "EDSR_x2.pb"  # Certifique-se de ter o modelo baixado
# sr.readModel(model_path)
# sr.setModel("edsr", 2)

# # Pré-processar x_train com upscaling
# upscaled_train = []
# for i in range(x_train.shape[0]):
#     img_bgr = cv2.cvtColor(x_train[i], cv2.COLOR_RGB2BGR)
#     img_upscaled = sr.upsample(img_bgr)  # Aplicar o modelo SR para upscaling
#     upscaled_train.append(cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2RGB))

# # Pré-processar x_test com upscaling
# upscaled_test = []
# for i in range(x_test.shape[0]):
#     img_bgr = cv2.cvtColor(x_test[i], cv2.COLOR_RGB2BGR)
#     img_upscaled = sr.upsample(img_bgr)  # Aplicar o modelo SR para upscaling
#     upscaled_test.append(cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2RGB))

# # Converter para numpy array
# upscaled_train = np.array(upscaled_train)
# upscaled_test = np.array(upscaled_test)

# # Normalizar as imagens para o intervalo [-1, 1]
# x_train, x_test = (upscaled_train / 255.0 - 0.5) * 2, (upscaled_test / 255.0 - 0.5) * 2

# # Plotar as primeiras 25 imagens originais
# fig, axes = plt.subplots(5, 5, figsize=(15, 7))
# axes = axes.flatten()

# for i in range(25):
#     axes[i].imshow((x_train[i] / 2 + 0.5))  # Desnormalizar para exibição
#     axes[i].set_title(f'Upscaled {i+1}')
#     axes[i].axis('off')

# plt.tight_layout()
# plt.show()

# # Criar o modelo
# model = models.Sequential([
#     layers.Input(shape=(x_train.shape[1], x_train.shape[2], 3)),

#     # Primeiro bloco convolucional
#     layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2, 2)),
#     layers.Dropout(0.2),

#     # Segundo bloco convolucional
#     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2, 2)),
#     layers.Dropout(0.2),

#     # Terceiro bloco convolucional
#     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2, 2)),
#     layers.Dropout(0.2),

#     # Camadas densas
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.2),
#     layers.Dense(10, activation='softmax')  # Saída para 10 classes
# ])

# # Compilar o modelo
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Treinar o modelo
# history = model.fit(x_train, y_train, epochs=25, batch_size=32, validation_data=(x_test, y_test))

# # Avaliar o modelo
# test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
# print(f"Test accuracy: {test_acc}")

# # Plotar histórico de treinamento
# plt.figure(figsize=(10, 4))

# # Acurácia
# plt.subplot(1, 1, 1)
# plt.plot(history.history['accuracy'], '.-', label='Train accuracy')
# plt.plot(history.history['val_accuracy'], '.-', label='Validation accuracy')
# plt.xlabel('Epochs')
# plt.legend()
# plt.grid()

# plt.show()

# # Salvar o modelo treinado
# model.save('v44.keras')







import numpy as np
import cv2
from cv2 import dnn_superres
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
import tensorflow as tf
import random

# Carregar o dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Exibir as dimensões dos dados
print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
print(f"x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}")

# Criar objeto SR para upscaling
print("Criando o objeto SR")
sr = dnn_superres.DnnSuperResImpl_create()
model_path = "EDSR_x2.pb"  # Certifique-se de ter o modelo baixado
sr.readModel(model_path)
sr.setModel("edsr", 2)

# Pré-processar x_train com upscaling
upscaled_train = []
for i in range(x_train.shape[0]):
    img_bgr = cv2.cvtColor(x_train[i], cv2.COLOR_RGB2BGR)
    img_upscaled = sr.upsample(img_bgr)  # Aplicar o modelo SR para upscaling
    upscaled_train.append(cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2RGB))

# Pré-processar x_test com upscaling
upscaled_test = []
for i in range(x_test.shape[0]):
    img_bgr = cv2.cvtColor(x_test[i], cv2.COLOR_RGB2BGR)
    img_upscaled = sr.upsample(img_bgr)  # Aplicar o modelo SR para upscaling
    upscaled_test.append(cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2RGB))

# Converter para numpy array
upscaled_train = np.array(upscaled_train)
upscaled_test = np.array(upscaled_test)

# Normalizar as imagens para o intervalo [-1, 1]
x_train, x_test = (upscaled_train / 255.0 - 0.5) * 2, (upscaled_test / 255.0 - 0.5) * 2

# Plotar as primeiras 25 imagens originais
fig, axes = plt.subplots(5, 5, figsize=(15, 7))
axes = axes.flatten()

for i in range(25):
    axes[i].imshow((x_train[i] / 2 + 0.5))  # Desnormalizar para exibição
    axes[i].set_title(f'Upscaled {i+1}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Criar o modelo
model = models.Sequential([
    layers.Input(shape=(x_train.shape[1], x_train.shape[2], 3)),

    # Primeiro bloco convolucional
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    # Segundo bloco convolucional
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    # Terceiro bloco convolucional
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    # Camadas densas
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')  # Saída para 10 classes
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(x_train, y_train, epochs=25, batch_size=32, validation_data=(x_test, y_test))

# Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Plotar histórico de treinamento
plt.figure(figsize=(10, 4))

# Acurácia
plt.subplot(1, 1, 1)
plt.plot(history.history['accuracy'], '.-', label='Train accuracy')
plt.plot(history.history['val_accuracy'], '.-', label='Validation accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.grid()

plt.show()

# Salvar o modelo treinado
model.save('v44.keras')

# Testar o modelo com 10 itens aleatórios da base de teste
random_indices = random.sample(range(len(x_test)), 10)
test_samples = x_test[random_indices]
test_labels = y_test[random_indices]

# Fazer previsões
predictions = model.predict(test_samples)

# Mostrar as imagens e as classificações
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.imshow((test_samples[i] / 2 + 0.5))  # Desnormalizar para exibição
    ax.set_title(f"Pred: {np.argmax(predictions[i])}, True: {test_labels[i][0]}")
    ax.axis('off')

plt.tight_layout()
plt.show()

# Carregar imagens externas e classificá-las
external_images = [
    'caxorro1(1).jpg', 'caxorro1(2).jpg', 'caxorro1(3).jpg',
    'caxorro2(1).jpg', 'caxorro2(2).jpg', 'caxorro2(3).jpg'
]

external_imgs_processed = []

for img_path in external_images:
    img = cv2.imread(img_path)  # Ler a imagem
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converter para RGB
    img_resized = cv2.resize(img_rgb, (x_train.shape[1], x_train.shape[2]))  # Redimensionar para o tamanho da entrada do modelo
    img_normalized = (img_resized / 255.0 - 0.5) * 2  # Normalizar
    external_imgs_processed.append(img_normalized)

external_imgs_processed = np.array(external_imgs_processed)

# Fazer previsões nas imagens externas
external_predictions = model.predict(external_imgs_processed)

# Mostrar as imagens externas e as previsões
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.imshow((external_imgs_processed[i] / 2 + 0.5))  # Desnormalizar para exibição
    ax.set_title(f"Prediction: {np.argmax(external_predictions[i])}")
    ax.axis('off')

plt.tight_layout()
plt.show()
