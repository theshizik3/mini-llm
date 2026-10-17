from src.datasets.pretraining.sft_pq_dataset import SFTParquetDataset

import json


class ZeroAgency(SFTParquetDataset):
    repo = "ZeroAgency/ru-tasks-conversation"
    columns = ["conversation"]

    def handler(self, row):
        res = ""
        for s in row:
            d = json.loads(s)
            res += " ".join([m["content"] for m in d])
        return res
