from pydantic import BaseModel


class ExtractDSConfig(BaseModel):
    name: str
    column: str
    limit: int
    ext: str
