from pydantic import BaseModel
from typing import Union


class DimensionConfiguration(BaseModel):
    name: str
    min: Union[float, int]
    max: Union[float, int]
    integers_only: bool = False
