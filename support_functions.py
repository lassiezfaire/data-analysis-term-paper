import random

def create_subset(X, y, fraction=1):
    dataset_size = len(X)
    subset_size = int(dataset_size * fraction)
    indices = random.sample(range(dataset_size), subset_size)
    return X[indices], y[indices]