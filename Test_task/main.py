import keras.models
from keras.datasets import mnist
from keras import utils
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import confusion_matrix

# Загрузка тестового датасета и подготовка его к использованию
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(10000, 784) / 255

y_test = utils.to_categorical(y_test, 10)

# Загрузка сохраненной модели
model = keras.models.load_model("mnist_dense.h5")

# Вычисление точности
scores = model.evaluate(x_test, y_test, verbose=0)

# Получение предсказаний для тестовых данных
y_pred = np.argmax(model.predict(x_test), axis=-1)

# Вывод части предсказаний
plt.figure(figsize=(12, 8))
plt.suptitle("Доля верных ответов на тестовых данных: " + str(round(scores[1] * 100, 4)) + "%")

for i in range(0, 50):
    choose = randint(0, 10000)
    plt.subplot(5, 10, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[choose].reshape(28, 28), cmap=plt.cm.binary)

    prediction = y_pred[choose]
    if prediction != np.argmax(y_test[choose]):
        plt.subplot(5, 10, i + 1).xaxis.label.set_color('red')
    else:
        plt.subplot(5, 10, i + 1).xaxis.label.set_color('green')
    plt.xlabel("prediction=" + str(prediction) + '\nactual=' + str(np.argmax(y_test[choose])))

plt.subplots_adjust(left=0.025, bottom=0.075, right=0.975, top=0.915)
plt.show()
