import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models, layers
from tensorflow.keras.utils import plot_model

def train_imdb_model():
    """
    Función para entrenar un modelo de clasificación binaria en el dataset IMDB.
    Retorna el modelo entrenado y los datos de prueba.
    """
    # Cargar los datos de IMDB
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    # Obtener el índice de palabras y decodificar una reseña de ejemplo
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

    print("Primera reseña decodificada:")
    print(decoded_review)
    print("Etiqueta de la primera reseña (1=Positiva, 0=Negativa):", train_labels[0])

    # Función para realizar one-hot encoding de las reseñas
    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    # Vectorizar datos
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    # Convertir etiquetas a flotantes
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    # Separar datos de validación
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    # Definir la arquitectura del modelo
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10000,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compilar el modelo
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    history = model.fit(
        partial_x_train, partial_y_train,
        epochs=20, batch_size=512,
        validation_data=(x_val, y_val)
    )

    # Evaluar el modelo en datos de prueba
    results = model.evaluate(x_test, y_test)
    print("\nResultados del modelo en el conjunto de prueba (Loss, Accuracy):", results)

    # Graficar pérdida
    history_dict = history.history
    plt.plot(range(1, len(history_dict['loss']) + 1), history_dict['loss'], 'bo', label='Training loss')
    plt.plot(range(1, len(history_dict['val_loss']) + 1), history_dict['val_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Graficar precisión
    plt.clf()
    plt.plot(range(1, len(history_dict['accuracy']) + 1), history_dict['accuracy'], 'bo', label='Training accuracy')
    plt.plot(range(1, len(history_dict['val_accuracy']) + 1), history_dict['val_accuracy'], 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Retornar el modelo y los datos de prueba
    return model, x_test, y_test

if __name__ == "__main__":
    model, x_test, y_test = train_imdb_model()
    print("\nEjemplo de predicción:")
    print(model.predict(x_test[:2]))
