from src.datasets.pretraining.sft_pq_dataset import SFTParquetDataset


class Kolmogorov(SFTParquetDataset):
    repo = "attn-signs/kolmogorov-3"
    columns = ["problem", "solution"]
        
    def handler(self, row):
        return " ".join(row)
        