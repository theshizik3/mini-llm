from pydantic import BaseModel


class FetchDatasetConfig(BaseModel):
    name: str
    pat: str
    files: int = 0
    random: bool = False
    