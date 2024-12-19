import random

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

from keras import datasets
from keras import utils

from active_learning import least_confidence_sampling, badge_sampling, bald_sampling
from model import Model
from support_functions import create_subset

mnist = datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train_categorical = utils.to_categorical(y_train, 10)
y_test_categorical = utils.to_categorical(y_test, 10)

model = Model()

model.fit(X_train, y_train_categorical, epochs=10, batch_size=32, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

f1_full  = f1_score(y_test, y_pred_classes, average='weighted')
print(f'F1 Score: {f1_full}')

f1_scores_dict = {}
fractions = [0.01, 0.1, 0.2]

# Случаное обучение
for fraction in fractions:
    X_train_subset, y_train_subset = create_subset(X_train, y_train_categorical, fraction)

    model = Model()

    model.fit(X_train_subset, y_train_subset, epochs=5, batch_size=128, validation_split=0.2)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    f1_scores_dict[f'Random {int(fraction*100)}%'] = f1
    print(f'Fraction: {fraction*100}%, F1 Score: {f1}')

# Активное обучение
for fraction in fractions:
    initial_size = int(len(X_train) * 0.01)
    labeled_indices = random.sample(range(len(X_train)), initial_size)
    unlabeled_indices = list(set(range(len(X_train))) - set(labeled_indices))

    X_labeled, y_labeled = X_train[labeled_indices], y_train_categorical[labeled_indices]
    X_unlabeled = X_train[unlabeled_indices]

    total_acquisition = int(len(X_train) * fraction) - initial_size
    acquisition_iterations = total_acquisition // initial_size

    # Активное обучение с эвристикой Least Confident
    lc_labeled_indices = labeled_indices.copy()
    lc_unlabeled_indices = unlabeled_indices.copy()
    lc_model = Model()
    lc_model.fit(X_labeled, y_labeled, epochs=5, batch_size=128, validation_split=0.2)

    for _ in range(acquisition_iterations):
        new_indices = least_confidence_sampling(lc_model, X_train[lc_unlabeled_indices], initial_size)
        new_data_indices = [lc_unlabeled_indices[i] for i in new_indices]
        lc_labeled_indices.extend(new_data_indices)
        lc_unlabeled_indices = list(set(lc_unlabeled_indices) - set(new_data_indices))
        X_labeled, y_labeled = X_train[lc_labeled_indices], y_train_categorical[lc_labeled_indices]
        lc_model.fit(X_labeled, y_labeled, epochs=5, batch_size=128, validation_split=0.2)

    y_pred = lc_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    f1_lc = f1_score(y_test, y_pred_classes, average='weighted')
    f1_scores_dict[f'Least Confident {int(fraction * 100)}%'] = f1_lc
    print(f'Fraction: {fraction*100}%, F1 Score (Least Confident): {f1_lc}')

    # Активное обучение с эвристикой BALD
    bald_labeled_indices = labeled_indices.copy()
    bald_unlabeled_indices = unlabeled_indices.copy()
    bald_model = Model()
    bald_model.fit(X_labeled, y_labeled, epochs=5, batch_size=128, validation_split=0.2)

    for _ in range(acquisition_iterations):
        new_indices = bald_sampling(bald_model, X_train[bald_unlabeled_indices], initial_size)
        new_data_indices = [bald_unlabeled_indices[i] for i in new_indices]
        bald_labeled_indices.extend(new_data_indices)
        bald_unlabeled_indices = list(set(bald_unlabeled_indices) - set(new_data_indices))
        X_labeled, y_labeled = X_train[bald_labeled_indices], y_train_categorical[bald_labeled_indices]
        bald_model.fit(X_labeled, y_labeled, epochs=5, batch_size=128, validation_split=0.2)

    y_pred = bald_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    f1_bald = f1_score(y_test, y_pred_classes, average='weighted')
    f1_scores_dict[f'BALD {int(fraction * 100)}%'] = f1_bald
    print(f'Fraction: {fraction*100}%, F1 Score (BALD): {f1_bald}')

    # Активное обучение с эвристикой BADGE
    badge_labeled_indices = labeled_indices.copy()
    badge_unlabeled_indices = unlabeled_indices.copy()
    badge_model = Model()
    badge_model.fit(X_labeled, y_labeled, epochs=5, batch_size=128, validation_split=0.2)

    for _ in range(acquisition_iterations):
        new_indices = badge_sampling(badge_model, X_train[badge_unlabeled_indices], initial_size)
        new_data_indices = [badge_unlabeled_indices[i] for i in new_indices]
        badge_labeled_indices.extend(new_data_indices)
        badge_unlabeled_indices = list(set(badge_unlabeled_indices) - set(new_data_indices))
        X_labeled, y_labeled = X_train[badge_labeled_indices], y_train_categorical[badge_labeled_indices]
        badge_model.fit(X_labeled, y_labeled, epochs=5, batch_size=128, validation_split=0.2)

    y_pred = badge_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    f1_badge = f1_score(y_test, y_pred_classes, average='weighted')
    f1_scores_dict[f'BADGE {int(fraction * 100)}%'] = f1_badge
    print(f'Fraction: {fraction*100}%, F1 Score (BADGE): {f1_badge}')

fractions_labels = [f'{int(f*100)}%' for f in fractions]
random_f1_scores = [f1_scores_dict[f'Random {int(f*100)}%'] for f in fractions]
least_confident_f1_scores = [f1_scores_dict[f'Least Confident {int(f*100)}%'] for f in fractions]
bald_f1_scores = [f1_scores_dict[f'BALD {int(f*100)}%'] for f in fractions]
badge_f1_scores = [f1_scores_dict[f'BADGE {int(f*100)}%'] for f in fractions]

plt.plot(fractions_labels, random_f1_scores, marker='o', linestyle='-', color='b', label='Random')
plt.plot(fractions_labels, least_confident_f1_scores, marker='o', linestyle='-', color='g', label='Least Confident')
plt.plot(fractions_labels, bald_f1_scores, marker='o', linestyle='-', color='m', label='BALD')
plt.plot(fractions_labels, badge_f1_scores, marker='o', linestyle='-', color='c', label='BADGE')
plt.axhline(y=f1_full, color='r', linestyle='--', label='Full Data Model')

plt.xlabel('Percentage of Training Data')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Percentage of Training Data')
plt.legend()
plt.grid(True)
plt.show()
