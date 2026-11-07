import numpy as np

from src.readers.reader import Reader


class BlockBinaryReader(Reader):
    def __init__(self, path, seq_len, dtype=np.int32):
        self.seq_len = seq_len
        self.dtype = dtype
        self.path = path
    
    def __iter__(self):
        arr = np.memmap(self.path, dtype=self.dtype, mode="r")
        for i in range(0, len(arr) - self.seq_len + 1, self.seq_len):
            yield np.array(arr[i:i + self.seq_len])
        del arr
