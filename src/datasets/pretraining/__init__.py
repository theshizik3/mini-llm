from src.datasets.pretraining.culturax import CulturaX
from src.datasets.pretraining.multi_pq_dataset import MultiParquetDataset
from src.datasets.pretraining.taiga_proza import TaigaProza
from src.datasets.pretraining.taiga_stihi import TaigaStihi
from src.datasets.pretraining.wikipedia import Wikipedia
from src.datasets.pretraining.math_d0rj import MathD0rj
from src.datasets.pretraining.kolmogorov import Kolmogorov
from src.datasets.pretraining.math_zeroagency import ZeroAgency


__all__ = [
    "CulturaX",
    "TaigaProza",
    "MultiParquetDataset",
    "TaigaStihi",
    "Wikipedia",
    "MathD0rj",
    "Kolmogorov",
    "ZeroAgency"
]
