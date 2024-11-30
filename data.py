import numpy as np

from sklearn.model_selection import train_test_split


def train_test(data: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Обёртка для функции train_test_split, чтобы не замусоривать main

    :param data: признаки
    :param target: ответы
    :return: признаки тренировочной, признаки тестовой, ответы тренировочной, ответы тестовой
    """

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.25, random_state=None)

    return X_train, X_test, y_train, y_test
