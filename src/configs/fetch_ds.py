from pydantic import BaseModel


class FetchDSConfig(BaseModel):
    name: str
    config: str = ""
    n_files: int | None = None
    