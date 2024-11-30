import numpy as np
from tensorflow.keras.models import Sequential

from active_learning.base_strategy import BaseStrategy

class LeastConfidenceSampling(BaseStrategy):
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

    def active_learning(self, num_instances: int) -> tuple[np.ndarray, np.ndarray]:
        probs = self.model.predict(self.X_train)
        uncertainties = 1 - np.max(probs, axis=1)
        selected_indices = np.argsort(uncertainties)[-num_instances:]
        return self.X_train[selected_indices], self.y_train[selected_indices]
