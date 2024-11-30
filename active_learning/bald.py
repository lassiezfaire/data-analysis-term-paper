import numpy as np
from tensorflow.keras.models import Sequential

from active_learning.base_strategy import BaseStrategy

class BaldSampling(BaseStrategy):
    def __init__(
            self,
            model: Sequential,
            X_train: np.ndarray,
            y_train: np.ndarray,
            n_dropout_samples: int = 10
    ):

        super().__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.n_dropout_samples = n_dropout_samples

    def active_learning(self, num_instances: int) -> tuple[np.ndarray, np.ndarray]:
        probs = np.array([self.model.predict(self.X_train, verbose=0) for _ in range(self.n_dropout_samples)])
        mean_probs = np.mean(probs, axis=0)
        uncertainties = np.sum(probs * (np.log(probs) - np.log(mean_probs)), axis=2).sum(axis=0)
        selected_indices = np.argsort(uncertainties)[-num_instances:]
        return self.X_train[selected_indices], self.y_train[selected_indices]
