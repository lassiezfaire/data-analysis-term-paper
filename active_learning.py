import numpy as np
import tensorflow as tf

def least_confidence_sampling(model, X_unlabeled, num_samples):
    y_pred = model.predict(X_unlabeled)
    confidences = np.max(y_pred, axis=1)
    least_confident_indices = np.argsort(confidences)[:num_samples]
    return least_confident_indices

def bald_sampling(model, X_unlabeled, num_samples, num_dropout_samples=10):
    predictions = []
    for _ in range(num_dropout_samples):
        y_pred = model.predict(X_unlabeled, verbose=0)
        predictions.append(y_pred)
    predictions = np.array(predictions)
    mean_predictions = np.mean(predictions, axis=0)
    entropy_mean = -np.sum(mean_predictions * np.log(mean_predictions + 1e-10), axis=1)
    entropy_individual = -np.sum(predictions * np.log(predictions + 1e-10), axis=2)
    mean_entropy = np.mean(entropy_individual, axis=0)
    bald_scores = entropy_mean - mean_entropy
    bald_indices = np.argsort(bald_scores)[-num_samples:]
    return bald_indices

def badge_sampling(model, X_unlabeled, num_samples):
    X_unlabeled_tensor = tf.convert_to_tensor(X_unlabeled, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(X_unlabeled_tensor)
        y_pred = model(X_unlabeled_tensor, training=True)
    gradients = tape.gradient(y_pred, X_unlabeled_tensor)

    gradient_norms = np.linalg.norm(gradients.numpy(), axis=(1, 2))

    badge_indices = np.argsort(gradient_norms)[-num_samples:]
    return badge_indices
