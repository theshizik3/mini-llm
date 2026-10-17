from src.datasets.pretraining.sft_pq_dataset import SFTParquetDataset

import re


class MathD0rj(SFTParquetDataset):
    repo = "d0rj/mathematics_dataset"
    columns = ["question", "answer"]
    
    def __init__(self, token, max_files=None, shuffle=False, langs=("ru",)):
        self.pat = r"^data/(?:%s)-\d{5}-of-\d{5}\.parquet$" % (
            "|".join(re.escape(l) for l in langs)
        )
        super().__init__(token, max_files, shuffle)
    
    def handler(self, row):
            return " ".join(row)