import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_model(num_features: int) -> Sequential:
    """Функция, создающая последовательную (sequential) модель

    :param num_features: количество признаков у объектов
    :return: созданная модель
    """

    model = Sequential(
        [
            Dense(64, input_dim=num_features, activation='relu'),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ]
    )
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test) -> np.float64:
    """Функция тренировки и оценки модели

    :param model: модель
    :param X_train: признаки тренировочной выборки
    :param X_test: признаки тестовой выборки
    :param y_train: ответы тренировочной выборки
    :param y_test: ответы тестовой выборки
    :return: f-мера обученной модели
    """

    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        verbose=0
        )

    y_pred = np.argmax(model.predict(X_test), axis=1)
    return f1_score(y_test, y_pred, average='weighted')
