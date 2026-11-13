from pydantic import BaseModel


class FetchDSConfig(BaseModel):
    name: str
    pat: str
    n_files: int | None = None
    