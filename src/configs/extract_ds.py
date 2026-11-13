from pydantic import BaseModel


class ExtractDSConfig(BaseModel):
    name: str
    pat: str
    column: str
    limit: int
