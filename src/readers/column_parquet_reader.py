import pyarrow.parquet as pq

from src.readers.reader import Reader


class ColumnParquetReader(Reader):
    def __init__(self, path, batch_size, column):
        self.batch_size = batch_size
        self.column = column
        self.path = path
    
    def __iter__(self):
        parquet_file = pq.ParquetFile(self.path)
        for batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=[self.column]):
            column = batch.column(0)
            for row in column.to_pylist():
                yield row
        parquet_file.close()
