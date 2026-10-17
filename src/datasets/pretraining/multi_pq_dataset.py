import random


class MultiParquetDataset:
    def __init__(self, *args, shuffle=False, weights=None):
        self.datasets = list(args)
        self.shuffle = shuffle
        if weights is None:
            weights = [1.0] * len(self.datasets)
        self.weights = list(weights)

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)

    def __iter__(self):
        if self.shuffle:
            iters = [iter(ds) for ds in self.datasets]
            weights = list(self.weights)

            while iters:
                i = random.choices(range(len(iters)), weights=weights)[0]
                try:
                    yield next(iters[i])
                except StopIteration:
                    iters.pop(i)
                    weights.pop(i)
        else:
            iters = list(self.datasets)
            for it in iters:
                for s in it:
                    yield s
