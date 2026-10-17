import random


class MultiSFTDataset:
    def __init__(self, *args, shuffle=False,):
        self.samples = []
        for ds in args:
            self.samples.extend(ds.samples)
        if shuffle:
            random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __iter__(self):
        for m in self.samples:
            yield m        
