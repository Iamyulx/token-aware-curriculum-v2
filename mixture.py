import random

class DatasetMixture:
    def __init__(self, datasets, weights):
        self.datasets = datasets
        self.weights = weights

    def sample(self):
        dataset = random.choices(self.datasets, weights=self.weights)[0]
        idx = random.randint(0, len(dataset) - 1)
        return dataset[idx]
