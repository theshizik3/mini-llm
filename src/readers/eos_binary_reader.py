import numpy as np

from src.readers.reader import Reader


class EOSBinaryReader(Reader):
    def __init__(self, path, eos, dtype=np.int32):
        self.eos = eos
        self.dtype = dtype
        self.path = path
    
    def __iter__(self):
        arr = np.memmap(self.path, dtype=self.dtype, mode="r")
        hits = np.flatnonzero(arr == self.eos)
        starts = np.concatenate([[0], (hits + 1)[:-1]])
        ends = hits + 1
        for start, end in zip(starts, ends):
            yield np.asarray(arr[start:end])
