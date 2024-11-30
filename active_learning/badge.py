import numpy as np
from tensorflow.keras.models import Sequential

from active_learning.base_strategy import BaseStrategy


class BadgeSampling(BaseStrategy):
    def __init__(
            self,
            model: Sequential,
            X_train: np.ndarray,
            y_train: np.ndarray
    ):
        super().__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    def active_learning(self, num_instances) -> tuple[np.ndarray, np.ndarray]:
        predictions = self.model.predict(self.X_train)
        gradients = np.gradient(predictions, axis=1)
        gradient_norms = np.linalg.norm(gradients, axis=1)
        selected_indices = np.argsort(gradient_norms)[-num_instances:]
        return self.X_train[selected_indices], self.y_train[selected_indices]
