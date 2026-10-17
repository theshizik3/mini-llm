from src.datasets.pretraining.pq_dataset import ParquetDataset


class TaigaStihi(ParquetDataset):
    repo = "cointegrated/taiga_stripped_stihi"
    column = "text"
