from src.datasets.pretraining.pq_dataset import ParquetDataset


class CulturaX(ParquetDataset):
    repo = "uonlp/CulturaX"
    column = "text"

    def __init__(self, token, max_files=None, shuffle=False, langs=("ru",)):
        self.pat = r"^(?:%s)/(?:%s)_part_\d+\.parquet$" % (
            "|".join(langs), "|".join(langs)
        )
        super().__init__(token, max_files, shuffle)
