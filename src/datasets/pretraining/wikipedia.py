from src.datasets.pretraining.pq_dataset import ParquetDataset

import re


class Wikipedia(ParquetDataset):
    repo = "wikimedia/wikipedia"
    column = "text"
        
    def __init__(self, token, max_files=None, shuffle=False, langs=("ru",)):
        self.pat = r"^\d{8}\.(?:%s)/train-\d{5}-of-\d{5}\.parquet$" % (
            "|".join(re.escape(l) for l in langs)
        )
        super().__init__(token=token, max_files=max_files, shuffle=shuffle)
        