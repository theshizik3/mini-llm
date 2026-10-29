from src.datasets.pretraining.zst_jsonl_dataset import ZstJsonlDataset


class Wikireading(ZstJsonlDataset):
    column = "text"
    repo = "its5Q/wikireading"
