from src.datasets.pretraining.process_pq_dataset import ProcessParquetDataset


class Kolmogorov(ProcessParquetDataset):
    repo = "attn-signs/kolmogorov-3"
    columns = ["problem", "solution"]
        
    def handler(self, row):
        return " ".join(row)
        