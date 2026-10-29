from src.datasets.pretraining.process_pq_dataset import ProcessParquetDataset

import json


class ZeroAgency(ProcessParquetDataset):
    repo = "ZeroAgency/ru-tasks-conversation"
    columns = ["conversation"]

    def handler(self, row):
        res = ""
        for s in row:
            d = json.loads(s)
            res += " ".join([m["content"] for m in d])
        return res
