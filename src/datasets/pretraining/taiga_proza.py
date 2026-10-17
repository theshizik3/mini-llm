from src.datasets.pretraining.pq_dataset import ParquetDataset


class TaigaProza(ParquetDataset):
    repo = "danasone/taiga_proza"
    column = "text"
