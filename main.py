import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

from data import train_test
from model import create_model, train_and_evaluate_model

from active_learning.least_confidence import LeastConfidenceSampling
from active_learning.bald import BaldSampling
from active_learning.badge import BadgeSampling

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test(data=X, target=y)

model = create_model(num_features=X_train.shape[1])

results = []

strategies = {
    'least_confident': LeastConfidenceSampling(model, X_train, y_train),
    'basd': BaldSampling(model, X_train, y_train),
    'badge': BadgeSampling(model, X_train, y_train)
}

for strategy_name in strategies.keys():
    f1_scores = []
    for _ in range(5):
        model = create_model(num_features=X_train.shape[1])
        f1 = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
        f1_scores.append(f1)
    results.append((strategy_name, 'Full', np.mean(f1_scores)))

subset_sizes = [0.01, 0.1, 0.2]
for size in subset_sizes:
    n_instances = int(size * len(X_train))
    for strategy_name, cls_instance in strategies.items():
        f1_scores = []
        for _ in range(5):
            model = create_model(num_features=X_train.shape[1])
            X_train_subset, y_train_subset = cls_instance.active_learning(n_instances)
            f1 = train_and_evaluate_model(model, X_train_subset, X_test, y_train_subset, y_test)
            f1_scores.append(f1)
        results.append((strategy_name, f'{int(size*100)}%', np.mean(f1_scores)))


df_results = pd.DataFrame(results, columns=['Strategy', 'Subset', 'F1 Score'])
df_results.boxplot(column='F1 Score', by=['Strategy', 'Subset'])
plt.suptitle('')
plt.title('Comparison of F1 Scores for Different Training Subsets and Strategies')
plt.xlabel('Training Subset and Strategy')
plt.ylabel('F1 Score')
plt.xticks(rotation=45)

# Сохранение графика в файл
plt.savefig('images//f1_scores_comparison.png')
plt.show()

print(results)
