from src.readers.reader import Reader
from pathlib import Path

import random


class DirReader(Reader):
    def __init__(
        self, reader, dir, pat="*", shuffle=False,
        shards_in_buffer=4, buffer_len=10, **reader_kwargs
    ):
        dir = Path(dir)
        self.shards = list(dir.rglob(pat))
        self.shuffle = shuffle
        self.reader_kwargs = reader_kwargs
        self.shards_in_buffer = shards_in_buffer
        self.buffer_len = buffer_len
        self.reader = reader
    
    def __iter__(self):
        buffer = []
        shards = self.shards[:]
        if self.shuffle:
            iters = [iter(self.reader(shard, **self.reader_kwargs)) for shard in shards[:self.shards_in_buffer]]
            shards = shards[self.shards_in_buffer:]
            while iters:
                if len(buffer) < self.buffer_len:
                    try:
                        i = random.randint(0, len(iters)-1)
                        buffer.append(next(iters[i]))
                    except StopIteration:
                        iters.pop(i)
                        if shards:
                            iters.append(iter(self.reader(shards[0], **self.reader_kwargs)))
                            shards = shards[1:]
                    continue
                i = random.randint(0, len(buffer) - 1)
                yield buffer[i]
                buffer.pop(i)
            if buffer:
                yield from buffer
        else:
            for shard in self.shards:
                for row in self.reader(shard, **self.reader_kwargs):
                    yield row
        