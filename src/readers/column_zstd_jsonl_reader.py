from src.readers.reader import Reader

from zstandard import ZstdDecompressor

import io
import json


class ColumnZstdJsonlReader(Reader):
    def __init__(self, path, column):
        self.path = path
        self.column = column
    
    def __iter__(self):
        dctx = ZstdDecompressor()
        with open(self.path, 'rb') as f:
            with dctx.stream_reader(f) as reader:
                text_reader = io.TextIOWrapper(reader, encoding='utf-8')
                for line in text_reader:
                    yield json.loads(line)[self.column]
